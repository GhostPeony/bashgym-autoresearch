import pytest

from bashgym_autoresearch import __version__
from bashgym_autoresearch.cli import main


def test_version_flag_reports_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"bashgym-ar {__version__}"
