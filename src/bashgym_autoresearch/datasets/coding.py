"""Deterministic preparation of explicitly selected, pinned coding data.

These pure transforms do not fetch data or execute source code. Callers must
verify source file bytes against the pinned Hub revision before supplying rows.
The evaluation projection contains no reference answers. Canary references are
returned separately and must never be registered as the evaluation dataset.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MBPP_SOURCE = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
HUMANEVAL_PLUS_SOURCE = "evalplus/humanevalplus"
HUMANEVAL_PLUS_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
SFT_SOURCE = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
SFT_REVISION = "356bb069eee815daa6e23e9a282eeefe1490ad44"


@dataclass(frozen=True)
class PreparedBenchmark:
    tasks: list[dict[str, Any]]
    canaries: list[dict[str, Any]]


def encode_jsonl(rows: Sequence[Mapping], *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    """Canonical local bytes; hash these bytes when registering the transformed file."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("coding_jsonl_byte_limit")
    output = bytearray()
    for row in rows:
        line = (
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        if len(output) + len(line) > max_bytes:
            raise ValueError("coding_jsonl_byte_limit")
        output.extend(line)
    return bytes(output)


def _text(row: Mapping, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 131072:
        raise ValueError(f"coding_field_invalid:{key}")
    return value


def _select(rows: Iterable[Mapping], ids: Sequence, *, key: str, id_type: type) -> list[Mapping]:
    if not ids or len(ids) > 10000 or any(type(value) is not id_type for value in ids):
        raise ValueError("coding_selection_invalid")
    if len(set(ids)) != len(ids):
        raise ValueError("coding_selection_duplicate")
    wanted, selected, seen = set(ids), {}, set()
    for row in rows:
        value = row.get(key)
        if type(value) is not id_type:
            raise ValueError("coding_source_id_invalid")
        if value in seen:
            raise ValueError("coding_source_id_duplicate")
        seen.add(value)
        if value in wanted:
            selected[value] = row
    if set(selected) != wanted:
        raise ValueError("coding_selected_id_missing")
    return [selected[value] for value in ids]


def _imports(nodes: Sequence[ast.stmt], allowed: set[str]) -> list[ast.stmt]:
    imports = []
    for node in nodes:
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module.split(".")[0]]
            if any(alias.name == "*" for alias in node.names):
                raise ValueError("coding_star_import_unsupported")
        else:
            raise ValueError("coding_import_statement_invalid")
        if not set(modules).issubset(allowed):
            raise ValueError("coding_import_not_allowed")
        imports.append(node)
    return imports


def prepare_mbpp_sanitized(
    rows: Iterable[Mapping],
    *,
    task_ids: Sequence[int],
    split: str,
    allowed_imports: Sequence[str] = (),
) -> PreparedBenchmark:
    """Convert a fixed MBPP selection to HumanEval completion/check semantics.

    Import policy is explicit and empty by default. Only simple one-function
    references are supported; module globals, decorators, helper functions and
    computed defaults reject the selected task rather than changing its meaning.
    Select supported IDs before freezing the suite, never after model scoring.
    """
    if split not in {"train", "validation", "test", "prompt"}:
        raise ValueError("coding_mbpp_split_invalid")
    allowed = set(allowed_imports)
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in allowed):
        raise ValueError("coding_import_policy_invalid")
    tasks, canaries = [], []
    for row in _select(rows, task_ids, key="task_id", id_type=int):
        tree = ast.parse(_text(row, "code"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError("coding_mbpp_single_function_required")
        function = functions[0]
        if function.name == "check":
            raise ValueError("coding_mbpp_checker_name_collision")
        if function.decorator_list:
            raise ValueError("coding_mbpp_decorator_unsupported")
        import_nodes = [node for node in tree.body if node is not function]
        _imports(import_nodes, allowed)
        for value in [*function.args.defaults, *function.args.kw_defaults]:
            if value is not None:
                try:
                    ast.literal_eval(value)
                except (ValueError, TypeError):
                    raise ValueError("coding_mbpp_computed_default_unsupported") from None
        test_imports = row.get("test_imports")
        tests = row.get("test_list")
        if (
            not isinstance(test_imports, list)
            or not isinstance(tests, list)
            or not tests
            or any(not isinstance(value, str) for value in test_imports + tests)
        ):
            raise ValueError("coding_mbpp_tests_invalid")
        check_imports = _imports(
            [node for value in test_imports for node in ast.parse(value).body], allowed
        )
        assertions = [node for value in tests for node in ast.parse(value).body]
        if not assertions or any(not isinstance(node, ast.Assert) for node in assertions):
            raise ValueError("coding_mbpp_assertions_required")
        if not any(
            isinstance(node, ast.Name) and node.id == function.name
            for assertion in assertions
            for node in ast.walk(assertion)
        ):
            raise ValueError("coding_mbpp_entry_point_not_tested")

        stub = copy.deepcopy(function)
        # Replace the entire body, including original docstring. Only the task
        # description, callable signature and declared imports become model input.
        stub.body = [ast.Expr(value=ast.Constant(value=_text(row, "prompt")))]
        prompt = "\n".join(ast.unparse(node) for node in [*import_nodes, stub]) + "\n"
        check_body = [*check_imports]
        if function.name != "candidate":
            check_body.append(
                ast.Assign(
                    targets=[ast.Name(id=function.name, ctx=ast.Store())],
                    value=ast.Name(id="candidate", ctx=ast.Load()),
                )
            )
        check_body.extend(assertions)
        check = ast.FunctionDef(
            name="check",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="candidate")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=check_body,
            decorator_list=[],
        )
        test = ast.unparse(ast.fix_missing_locations(check)) + "\n"
        provenance = {"source": MBPP_SOURCE, "revision": MBPP_REVISION, "split": split}
        task_id = f"MBPP/{row['task_id']}"
        tasks.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "test": test,
                "entry_point": function.name,
                "provenance": provenance,
            }
        )
        completion = textwrap.indent("\n".join(ast.unparse(node) for node in function.body), "    ")
        canaries.append({"task_id": task_id, "canonical_solution": completion + "\n"})
    return PreparedBenchmark(tasks, canaries)


def prepare_humaneval_plus(
    rows: Iterable[Mapping],
    *,
    task_ids: Sequence[str],
) -> PreparedBenchmark:
    """Project pinned HF HumanEval+ rows, not native EvalPlus differential inputs."""
    tasks, canaries = [], []
    for row in _select(rows, task_ids, key="task_id", id_type=str):
        entry_point = _text(row, "entry_point")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point):
            raise ValueError("coding_entry_point_invalid")
        prompt, test, answer = (_text(row, key) for key in ("prompt", "test", "canonical_solution"))
        tree = ast.parse(test)
        checks = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check"
        ]
        if (
            len(checks) != 1
            or len(checks[0].args.args) != 1
            # The pinned HF projection calls a separate assertion() helper.
            # Validate its published shape, not an imagined inline-only grader.
            or not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
            or not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == checks[0].args.args[0].arg
                for node in ast.walk(checks[0])
            )
        ):
            raise ValueError("coding_humaneval_check_invalid")
        # Check syntax and selected entry point without executing reference code.
        program = ast.parse(prompt + answer)
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == entry_point for node in program.body
        ):
            raise ValueError("coding_humaneval_entry_point_missing")
        tasks.append(
            {
                "task_id": row["task_id"],
                "prompt": prompt,
                "test": test,
                "entry_point": entry_point,
                "provenance": {
                    "source": HUMANEVAL_PLUS_SOURCE,
                    "revision": HUMANEVAL_PLUS_REVISION,
                    "split": "test",
                },
            }
        )
        canaries.append({"task_id": row["task_id"], "canonical_solution": answer})
    return PreparedBenchmark(tasks, canaries)


def prepare_sft_examples(
    rows: Iterable[Mapping],
    *,
    row_ids: Sequence[int],
    max_examples: int,
    count_tokens: Callable[[list[dict[str, str]]], int] | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Export complete selected demonstrations with source and grouping provenance.

    The optional counter must use the actual learner tokenizer/chat template.
    An over-limit selected example rejects the export; it is never truncated or
    silently removed. Omitting the counter leaves token eligibility unverified.
    """
    if type(max_examples) is not int or max_examples < 1 or len(row_ids) > max_examples:
        raise ValueError("coding_sft_example_limit")
    if (count_tokens is None) != (max_tokens is None):
        raise ValueError("coding_sft_token_policy_incomplete")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ValueError("coding_sft_token_limit_invalid")
    output = []
    for row in _select(rows, row_ids, key="id", id_type=int):
        instruction, response, seed = (
            _text(row, key) for key in ("instruction", "response", "seed")
        )
        group = _text(row, "sha1")
        if not re.fullmatch(r"[0-9a-f]{40}", group):
            raise ValueError("coding_sft_group_invalid")
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        if count_tokens is not None:
            count = count_tokens(messages)
            if type(count) is not int or count < 1 or count > max_tokens:
                raise ValueError("coding_sft_complete_example_token_limit")
        provenance = {
            "source": SFT_SOURCE,
            "revision": SFT_REVISION,
            "split": "train",
            "row_id": row["id"],
            "group_id": group,
            "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
            "row_sha256": hashlib.sha256(encode_jsonl([dict(row)])).hexdigest(),
        }
        output.append({"messages": messages, "provenance": provenance})
    return output
