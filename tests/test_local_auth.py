import os
import stat

import pytest

from elt_rl.local_auth import load_local_key, save_local_key


def test_hidden_local_file_is_private_and_preserves_other_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "")
    path = tmp_path / ".env"
    path.write_text("OTHER_SETTING=keep\nTINKER_API_KEY=old-test-value\n")
    save_local_key("test-only-not-a-real-key", path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text().count("TINKER_API_KEY=") == 1
    assert "OTHER_SETTING=keep" in path.read_text()
    assert not list(tmp_path.glob(".tinker-key-*"))
    assert load_local_key(path)
    assert os.environ["TINKER_API_KEY"] == "test-only-not-a-real-key"


def test_explicit_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "explicit-test-key")
    path = tmp_path / ".env"
    save_local_key("file-test-key", path)
    assert load_local_key(path)
    assert os.environ["TINKER_API_KEY"] == "explicit-test-key"


def test_missing_file_does_not_contact_a_service(tmp_path, monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    assert not load_local_key(tmp_path / ".env")


@pytest.mark.parametrize("key", ["", "x y", "x\ny"])
def test_invalid_input_is_rejected_without_writing(tmp_path, key):
    with pytest.raises(ValueError):
        save_local_key(key, tmp_path / ".env")
    assert not (tmp_path / ".env").exists()


def test_symlink_is_not_followed(tmp_path):
    target = tmp_path / "target"
    target.write_text("untouched")
    path = tmp_path / ".env"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        save_local_key("test-key", path)
    assert target.read_text() == "untouched"


def test_cli_configuration_never_prints_key(tmp_path, monkeypatch, capsys):
    import socket

    from elt_rl.cli import main

    def forbid_connection(*args, **kwargs):
        pytest.fail("Local credential configuration attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_connection)

    path = tmp_path / ".env"
    monkeypatch.setattr("elt_rl.local_auth.credential_file", lambda: path)
    monkeypatch.setattr("elt_rl.local_auth.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("elt_rl.local_auth.getpass.getpass", lambda _: "secret-test-value")
    monkeypatch.setattr("sys.argv", ["elt-rl", "configure-tinker"])
    assert main() == 0
    output = capsys.readouterr()
    assert "secret-test-value" not in output.out + output.err
    assert path.stat().st_mode & 0o777 == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
