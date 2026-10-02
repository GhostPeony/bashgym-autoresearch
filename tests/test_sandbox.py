import os
import threading
from types import SimpleNamespace

import pytest

from bashgym_autoresearch.sandbox import run_program, validate_image

IMAGE = "python@sha256:" + "a" * 64


class FakeContainer:
    def __init__(self, client, kwargs):
        self.client, self.kwargs = client, kwargs
        self.released = threading.Event()
        self.killed = self.removed = False

    def wait(self):
        if self.client.hang:
            self.released.wait(30)
        return {"StatusCode": self.client.exit_code}

    def kill(self):
        self.killed = True
        self.released.set()

    def logs(self, stdout, stderr, stream):
        data = self.client.stdout if stdout else self.client.stderr
        for index in range(0, len(data), 7):
            yield data[index : index + 7]

    def remove(self, **kwargs):
        self.removed = True


class FakeClient:
    def __init__(self, *, exit_code=0, stdout=b"", stderr=b"", hang=False, missing=False):
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr
        self.hang, self.missing = hang, missing
        self.containers = SimpleNamespace(run=self.run)
        self.images = SimpleNamespace(get=self.get)
        self.created = []
        self.closed = False

    def ping(self):
        return True

    def get(self, image):
        if self.missing:
            raise RuntimeError("no such image")

    def run(self, **kwargs):
        container = FakeContainer(self, kwargs)
        self.created.append(container)
        return container

    def close(self):
        self.closed = True


def test_container_is_isolated_and_removed(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    client = FakeClient(stdout=b"line one\n42\n")
    result = run_program(
        IMAGE,
        {"solve.py": "print(42)"},
        ["python", "/workspace/solve.py"],
        timeout=5,
        read_only_mounts={"/home/user/input": data},
        client=client,
    )
    assert (result.exit_code, result.timed_out, result.error) == (0, False, None)
    assert result.stdout.splitlines()[-1] == "42"
    container = client.created[0]
    options = container.kwargs
    assert options["network_mode"] == "none" and options["read_only"] is True
    assert options["user"] == "65534:65534" and options["cap_drop"] == ["ALL"]
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert all(mount["ReadOnly"] for mount in options["mounts"])
    assert {mount["Target"] for mount in options["mounts"]} == {"/workspace", "/home/user/input"}
    assert container.removed


def test_output_keeps_the_tail():
    client = FakeClient(stdout=b"x" * 5000 + b"\nfinal answer\n")
    result = run_program(IMAGE, {}, ["python"], timeout=5, max_output=64, client=client)
    assert len(result.stdout.encode()) <= 64 and result.stdout.endswith("final answer\n")


def test_timeout_kills_and_reports():
    client = FakeClient(hang=True)
    result = run_program(IMAGE, {}, ["python"], timeout=0.2, client=client)
    assert result.timed_out and result.exit_code is None and client.created[0].killed


def test_missing_image_is_reported_not_raised():
    result = run_program(IMAGE, {}, ["python"], timeout=5, client=FakeClient(missing=True))
    assert result.error and "image" in result.error and result.exit_code is None


@pytest.mark.parametrize("image", ["python:3.12", "latest", "python@sha256:abc"])
def test_images_must_be_pinned_by_digest(image):
    with pytest.raises(ValueError):
        validate_image(image)


@pytest.mark.parametrize("target", ["relative", "/workspace/x", "/", "/a/../b"])
def test_unsafe_mounts_and_file_names_are_rejected(tmp_path, target):
    with pytest.raises(ValueError):
        run_program(
            IMAGE,
            {},
            ["python"],
            timeout=5,
            read_only_mounts={target: tmp_path},
            client=FakeClient(),
        )
    with pytest.raises(ValueError):
        run_program(IMAGE, {"../escape.py": ""}, ["python"], timeout=5, client=FakeClient())


REAL_IMAGE = os.environ.get("BGAR_TEST_SANDBOX_IMAGE")


@pytest.mark.skipif(not REAL_IMAGE, reason="set BGAR_TEST_SANDBOX_IMAGE to a local image digest")
def test_real_docker_isolation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "table.csv").write_text("a\n1\n2\n")
    program = (
        "import os, socket\n"
        "print(sum(int(x) for x in open('table.csv').read().split()[1:]))\n"
        "try:\n    open('w.txt', 'w'); print('writable')\nexcept OSError:\n    print('readonly')\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); print('network')\n"
        "except OSError:\n    print('offline')\n"
    )
    result = run_program(
        REAL_IMAGE,
        {"solve.py": program},
        ["python", "/workspace/solve.py"],
        timeout=60,
        workdir="/home/user/input",
        read_only_mounts={"/home/user/input": data},
    )
    assert result.error is None and result.exit_code == 0, result.stderr
    assert result.stdout.split() == ["3", "readonly", "offline"]
    slow = run_program(REAL_IMAGE, {}, ["python", "-c", "import time; time.sleep(60)"], timeout=2)
    assert slow.timed_out
