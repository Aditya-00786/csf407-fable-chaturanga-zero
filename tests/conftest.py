import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))  # lets tests import reference.py

from chaturanga import WHITE, GameState, parse_square, square_name  # noqa: E402


def position(pieces, side=WHITE, **kwargs):
    return GameState.from_pieces(pieces, side, **kwargs)


def destinations(state, from_name):
    """Names of the squares the piece on `from_name` can legally move to."""
    fr = parse_square(from_name)
    return {square_name(m % 64) for m in state.legal_moves() if m // 64 == fr}


@pytest.fixture(autouse=True)
def no_real_cloud(monkeypatch, tmp_path_factory):
    """Tests never touch a real Google Drive, Kaggle account or server: no Drive here, and the
    Kaggle and server settings in folders of their own."""
    monkeypatch.setenv("CHATURANGA_DRIVE", "")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path_factory.mktemp("kaggle-config")))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # Nor a real server: the app's settings and SSH config in folders of their own.
    home = tmp_path_factory.mktemp("chaturanga-home")
    monkeypatch.setenv("CHATURANGA_HOME", str(home))
    monkeypatch.setenv("CHATURANGA_SSH_CONFIG", str(home / "no-ssh-config"))
