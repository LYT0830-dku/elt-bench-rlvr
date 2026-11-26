import base64
import json
import stat

import pytest

from elt_rl.snowflake_setup import prepare, read_config


def test_bootstrap_keeps_private_key_local_and_reuses_it(tmp_path):
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    paths = prepare("xy12345.ap-southeast-7.aws", tmp_path)
    key_path = tmp_path / ".snowflake" / "elt_rl_key.p8"
    original = key_path.read_bytes()
    private = serialization.load_pem_private_key(original, password=None)
    public = base64.b64encode(
        private.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    ).decode()
    sql = paths["bootstrap"].read_text()
    assert public in sql
    assert "PRIVATE KEY" not in sql
    assert "TYPE = SERVICE" in sql
    assert "GRANT USAGE, CREATE SCHEMA ON DATABASE ELT_RL_DB TO ROLE ELT_RL_ROLE" in sql
    assert "GRANT ROLE ACCOUNTADMIN" not in sql
    assert "INITIALLY_SUSPENDED = TRUE" in sql
    config = json.loads(paths["config"].read_text())
    assert config["private_key_file"] == str(key_path.resolve())
    assert config["role"] == "ELT_RL_ROLE"
    assert config["authenticator"] == "SNOWFLAKE_JWT"
    for path in (key_path, paths["config"]):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_path.parent.stat().st_mode) == 0o700
    prepare("xy12345.ap-southeast-7.aws", tmp_path)
    assert key_path.read_bytes() == original


def test_explicit_environment_wins_over_local_config(tmp_path, monkeypatch):
    path = tmp_path / "credentials.snowflake.json"
    path.write_text(json.dumps({"account": "local"}))
    monkeypatch.setattr("elt_rl.snowflake_setup.config_file", lambda: path)
    monkeypatch.delenv("ELT_SNOWFLAKE_CONFIG", raising=False)
    assert read_config()["account"] == "local"
    monkeypatch.setenv("ELT_SNOWFLAKE_CONFIG", json.dumps({"account": "explicit"}))
    assert read_config()["account"] == "explicit"
    monkeypatch.setenv("ELT_SNOWFLAKE_CONFIG", "[]")
    with pytest.raises(ValueError, match="JSON object"):
        read_config()


def test_setup_rejects_unsafe_account_and_symlink(tmp_path):
    pytest.importorskip("cryptography")
    with pytest.raises(ValueError):
        prepare("account'; DROP DATABASE existing; --", tmp_path)
    assert not (tmp_path / ".snowflake").exists()
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / ".snowflake").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        prepare("example", tmp_path)
    assert list(target.iterdir()) == []
