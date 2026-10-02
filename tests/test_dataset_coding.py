import json

import pytest

from bashgym_autoresearch.datasets.coding import (
    HUMANEVAL_PLUS_REVISION,
    MBPP_REVISION,
    SFT_REVISION,
    encode_jsonl,
    prepare_humaneval_plus,
    prepare_mbpp_sanitized,
    prepare_sft_examples,
)


def mbpp(task_id=7, **changes):
    row = {
        "task_id": task_id,
        "prompt": "Return the integer increased by one.",
        "code": "def increment(value: int) -> int:\n    return value + 1\n",
        "test_imports": [],
        "test_list": ["assert increment(3) == 4", "assert increment(-1) == 0"],
        "source_file": "authored-test-fixture",
    }
    return row | changes


def check_fixture(task, completion):
    # Only hand-authored harmless fixtures are executed here, never downloaded code.
    namespace = {}
    exec(task["prompt"] + completion + "\n" + task["test"], namespace)
    namespace["check"](namespace[task["entry_point"]])


def test_mbpp_good_bad_and_no_answer_leak():
    bundle = prepare_mbpp_sanitized([mbpp()], task_ids=[7], split="validation")
    task = bundle.tasks[0]
    assert set(task) == {"task_id", "prompt", "test", "entry_point", "provenance"}
    assert task["task_id"] == "MBPP/7"
    assert task["provenance"] == {
        "source": "google-research-datasets/mbpp",
        "revision": MBPP_REVISION,
        "split": "validation",
    }
    assert "Return the integer" in task["prompt"]
    assert "return value + 1" not in task["prompt"]
    assert "assert increment" not in task["prompt"]
    check_fixture(task, bundle.canaries[0]["canonical_solution"])
    with pytest.raises(AssertionError):
        check_fixture(task, "    return value - 1\n")
    assert "canonical_solution" not in encode_jsonl(bundle.tasks).decode()


def test_imports_and_literal_default_are_explicit_and_operational():
    row = mbpp(
        code="import math\ndef increment(value, extra=1):\n    return math.floor(value) + extra",
        test_imports=["import math"],
        test_list=["assert increment(math.pi) == 4"],
    )
    with pytest.raises(ValueError, match="import_not_allowed"):
        prepare_mbpp_sanitized([row], task_ids=[7], split="validation")
    bundle = prepare_mbpp_sanitized(
        [row], task_ids=[7], split="validation", allowed_imports=("math",)
    )
    check_fixture(bundle.tasks[0], bundle.canaries[0]["canonical_solution"])


@pytest.mark.parametrize(
    "code",
    [
        "OFFSET = 1\ndef increment(value):\n    return value + OFFSET",
        "@property\ndef increment(value):\n    return value + 1",
        "def increment(value=compute_secret()):\n    return value + 1",
        "def increment(value):\n    return value + 1\ndef other(value):\n    return value",
        "def check(value):\n    return value + 1",
    ],
)
def test_unsupported_canonical_shapes_reject_selected_task(code):
    with pytest.raises(ValueError):
        prepare_mbpp_sanitized([mbpp(code=code)], task_ids=[7], split="validation")


@pytest.mark.parametrize(
    "rows,ids",
    [
        ([mbpp()], [7, 8]),
        ([mbpp(), mbpp()], [7]),
        ([mbpp()], [7, 7]),
        ([mbpp()], []),
        ([mbpp()], [True]),
    ],
)
def test_selection_never_drops_missing_duplicate_or_empty_tasks(rows, ids):
    with pytest.raises(ValueError):
        prepare_mbpp_sanitized(rows, task_ids=ids, split="validation")


def test_explicit_task_order_is_preserved_and_unknown_split_rejected():
    bundle = prepare_mbpp_sanitized([mbpp(3), mbpp(7)], task_ids=[7, 3], split="validation")
    assert [row["task_id"] for row in bundle.tasks] == ["MBPP/7", "MBPP/3"]
    with pytest.raises(ValueError, match="split"):
        prepare_mbpp_sanitized([mbpp()], task_ids=[7], split="evaluation")


def test_humaneval_plus_schema_good_bad_and_separate_canary():
    row = {
        "task_id": "HumanEval/0",
        "prompt": 'def increment(value):\n    """Increase the integer by one."""\n',
        "canonical_solution": "    return value + 1\n",
        "entry_point": "increment",
        "test": "def check(candidate):\n    assert candidate(3) == 4\n",
    }
    bundle = prepare_humaneval_plus([row], task_ids=["HumanEval/0"])
    task = bundle.tasks[0]
    assert task["provenance"]["revision"] == HUMANEVAL_PLUS_REVISION
    assert task["test"] == row["test"]
    assert task["prompt"] == row["prompt"]
    assert "canonical_solution" not in task
    check_fixture(task, bundle.canaries[0]["canonical_solution"])
    with pytest.raises(AssertionError):
        check_fixture(task, "    return value - 1\n")
    with pytest.raises(ValueError, match="check"):
        prepare_humaneval_plus([row | {"test": "assert True"}], task_ids=["HumanEval/0"])
    with pytest.raises(ValueError, match="check"):
        prepare_humaneval_plus(
            [row | {"test": "def check(candidate):\n    pass"}], task_ids=["HumanEval/0"]
        )


def test_description_is_only_a_docstring_even_with_quotes_and_newlines():
    description = 'A description containing """ and a newline.\nreturn secret_answer'
    bundle = prepare_mbpp_sanitized([mbpp(prompt=description)], task_ids=[7], split="validation")
    check_fixture(bundle.tasks[0], bundle.canaries[0]["canonical_solution"])


def test_humaneval_plus_indirect_assertion_helper_matches_published_shape():
    row = {
        "task_id": "HumanEval/0",
        "prompt": 'def increment(value):\n    """Increase by one."""\n',
        "canonical_solution": "    return value + 1\n",
        "entry_point": "increment",
        "test": (
            "def assertion(actual, expected):\n"
            "    assert actual == expected\n\n"
            "def check(candidate):\n"
            "    assertion(candidate(3), 4)\n"
        ),
    }
    bundle = prepare_humaneval_plus([row], task_ids=["HumanEval/0"])
    check_fixture(bundle.tasks[0], bundle.canaries[0]["canonical_solution"])
    with pytest.raises(AssertionError):
        check_fixture(bundle.tasks[0], "    return value - 1\n")
    with pytest.raises(ValueError, match="check"):
        prepare_humaneval_plus(
            [row | {"test": row["test"].replace("candidate(3)", "4")}],
            task_ids=["HumanEval/0"],
        )


@pytest.mark.parametrize("test_list", [[], ["assert True"], ["increment(3)"]])
def test_mbpp_must_have_assertions_that_reference_the_entry_point(test_list):
    with pytest.raises(ValueError):
        prepare_mbpp_sanitized([mbpp(test_list=test_list)], task_ids=[7], split="validation")


def test_sft_instruction_mapping_provenance_and_complete_example_cutoff():
    rows = [
        {
            "id": 42,
            "instruction": "Increment the integer.",
            "response": "def increment(x): return x + 1",
            "prompt": "GENERATOR SCAFFOLD MUST NOT BE TRAINED",
            "seed": "def source(x): return x",
            "sha1": "a" * 40,
        }
    ]
    result = prepare_sft_examples(rows, row_ids=[42], max_examples=2)
    assert result[0]["messages"] == [
        {"role": "user", "content": rows[0]["instruction"]},
        {"role": "assistant", "content": rows[0]["response"]},
    ]
    assert "GENERATOR SCAFFOLD" not in json.dumps(result)
    assert result[0]["provenance"]["revision"] == SFT_REVISION
    assert result[0]["provenance"]["seed_sha256"]
    assert result[0]["provenance"]["group_id"] == "a" * 40
    with pytest.raises(ValueError, match="token_limit"):
        prepare_sft_examples(
            rows,
            row_ids=[42],
            max_examples=2,
            count_tokens=lambda messages: 50,
            max_tokens=49,
        )
    with pytest.raises(ValueError, match="example_limit"):
        prepare_sft_examples(rows, row_ids=[42], max_examples=0)


def test_jsonl_is_stable_and_bounded():
    assert encode_jsonl([{"b": 2, "a": 1}]) == b'{"a":1,"b":2}\n'
    with pytest.raises(ValueError, match="byte_limit"):
        encode_jsonl([{"long": "value"}], max_bytes=2)
