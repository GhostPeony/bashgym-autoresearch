import pytest

from bashgym_autoresearch.auth import (
    AuthError,
    Forbidden,
    Principal,
    authenticate,
    create_token,
    require_human,
)
from bashgym_autoresearch.store import Store


def test_token_round_trip_stores_only_a_hash(tmp_path):
    store = Store(tmp_path / "state.db")
    token = create_token(store, "agent", "claude-code")
    assert token.startswith("bgar_")
    assert authenticate(store, token) == Principal(role="agent", label="claude-code")
    stored = [row["hash"] for row in store.read("SELECT hash FROM tokens")]
    assert token not in stored and len(stored) == 1


@pytest.mark.parametrize("token", ["", "bgar_unknown", None])
def test_unknown_or_missing_tokens_fail(tmp_path, token):
    store = Store(tmp_path / "state.db")
    with pytest.raises(AuthError):
        authenticate(store, token)


def test_require_human_rejects_agents():
    require_human(Principal(role="human", label="owner"))
    with pytest.raises(Forbidden):
        require_human(Principal(role="agent", label="codex"))


def test_invalid_role_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        create_token(Store(tmp_path / "state.db"), "admin", "x")
