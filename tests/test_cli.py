import json
import os

import pytest

from bashgym_autoresearch import __version__, cli
from bashgym_autoresearch.api import open_home
from bashgym_autoresearch.auth import authenticate


def test_version_flag_reports_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"bashgym-ar {__version__}"


def test_init_creates_a_human_token_once(tmp_path, capsys):
    home = tmp_path / "home"
    assert cli.main(["--home", str(home), "init"]) == 0
    token = (home / "human.token").read_text()
    assert authenticate(open_home(home).store, token).role == "human"
    assert cli.main(["--home", str(home), "init"]) == 0
    assert "already initialized" in capsys.readouterr().out
    assert (home / "human.token").read_text() == token


def test_token_command_goes_through_the_api(monkeypatch, capsys):
    calls = []

    class FakeClient:
        def create_token(self, role, label):
            calls.append((role, label))
            return {"token": "bgar_x", "role": role, "label": label}

    monkeypatch.setattr(cli, "_client", lambda args: FakeClient())
    assert cli.main(["token", "agent", "--label", "codex"]) == 0
    assert calls == [("agent", "codex")]


def test_init_never_mints_a_second_human_token(tmp_path, capsys):
    home = tmp_path / "home"
    cli.main(["--home", str(home), "init"])
    (home / "human.token").unlink()
    cli.main(["--home", str(home), "init"])
    assert "already initialized" in capsys.readouterr().out
    assert not (home / "human.token").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_serve_refuses_a_world_readable_state_directory(tmp_path):
    home = tmp_path / "home"
    cli.main(["--home", str(home), "init"])
    home.chmod(0o755)
    with pytest.raises(SystemExit, match="chmod 700"):
        cli.main(["--home", str(home), "serve"])


def test_agent_verbs_print_json_and_report_api_errors(monkeypatch, capsys):
    from bashgym_autoresearch.client import ClientError

    class FakeClient:
        def brief(self, campaign_id):
            return {"campaign_id": campaign_id, "next_action": {"kind": "propose_baseline"}}

        def pause(self, campaign_id):
            raise ClientError(409, {"detail": "not running"})

    monkeypatch.setattr(cli, "_client", lambda args: FakeClient())
    assert cli.main(["brief", "cmp_1"]) == 0
    assert json.loads(capsys.readouterr().out)["campaign_id"] == "cmp_1"
    assert cli.main(["pause", "cmp_1"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == 409


def test_commands_needing_the_api_require_a_token(monkeypatch):
    monkeypatch.delenv("BGAR_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="BGAR_TOKEN"):
        cli.main(["brief", "cmp_1"])
