"""Prepare local key-pair credentials and a reviewable Snowflake bootstrap script."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path

from elt_rl.local_auth import credential_file


def config_file() -> Path:
    return credential_file().parent / "credentials.snowflake.json"


def read_config(credentials_env: str = "ELT_SNOWFLAKE_CONFIG") -> dict:
    """Explicit connector arguments take precedence over the local project file."""
    value = os.environ.get(credentials_env)
    if value:
        config = json.loads(value)
    elif credentials_env == "ELT_SNOWFLAKE_CONFIG" and config_file().is_file():
        config = json.loads(config_file().read_text())
    else:
        raise RuntimeError(
            "Snowflake is not configured; run elt-rl prepare-snowflake --account ..."
        )
    if not isinstance(config, dict):
        raise ValueError("Snowflake connector configuration must be a JSON object")
    return config


def _write_private(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError("Snowflake setup files must not be symbolic links")
    fd, temporary = tempfile.mkstemp(prefix=".snowflake-setup-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _public_key(key_path: Path) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
            raise ValueError("Existing Snowflake key must be an RSA key of at least 2048 bits")
        os.chmod(key_path, 0o600)
    else:
        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        _write_private(
            key_path,
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
    public_key = base64.b64encode(
        key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    ).decode("ascii")
    return public_key


def prepare(account: str, root: Path | None = None) -> dict[str, Path]:
    """Only write local files. Never contact Snowflake or replace an existing key."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", account):
        raise ValueError("Use the Snowflake connector account identifier, without a URL or spaces")
    root = root or credential_file().parent
    root.mkdir(parents=True, exist_ok=True)
    directory = root / ".snowflake"
    if directory.is_symlink():
        raise ValueError("Snowflake setup directory must not be a symbolic link")
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    key_path = directory / "elt_rl_key.p8"
    connection_path = root / "credentials.snowflake.json"
    bootstrap_path = directory / "bootstrap.sql"
    for path in (key_path, connection_path, bootstrap_path):
        if path.is_symlink():
            raise ValueError("Snowflake setup files must not be symbolic links")
    public_key = _public_key(key_path)
    config = {
        "account": account,
        "user": "ELT_RL_RUNNER",
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": str(key_path.resolve()),
        "database": "ELT_RL_DB",
        "warehouse": "ELT_RL_WH",
        "role": "ELT_RL_ROLE",
    }
    _write_private(connection_path, (json.dumps(config, indent=2) + "\n").encode())
    script = f"""-- Run once in Snowflake Workspaces as your account administrator.
-- This file contains only a PUBLIC key. The private key stays on this computer.
-- The runtime account can use only the experiment database and warehouse.
USE ROLE ACCOUNTADMIN;

CREATE ROLE IF NOT EXISTS ELT_RL_ROLE;
GRANT ROLE ELT_RL_ROLE TO ROLE SYSADMIN;

CREATE DATABASE IF NOT EXISTS ELT_RL_DB;
CREATE WAREHOUSE IF NOT EXISTS ELT_RL_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

GRANT USAGE, CREATE SCHEMA ON DATABASE ELT_RL_DB TO ROLE ELT_RL_ROLE;
GRANT USAGE ON WAREHOUSE ELT_RL_WH TO ROLE ELT_RL_ROLE;

CREATE USER IF NOT EXISTS ELT_RL_RUNNER
  TYPE = SERVICE
  DEFAULT_ROLE = ELT_RL_ROLE
  DEFAULT_WAREHOUSE = ELT_RL_WH
  COMMENT = 'ELT-Bench experimental runtime';
ALTER USER ELT_RL_RUNNER SET RSA_PUBLIC_KEY = '{public_key}';
GRANT ROLE ELT_RL_ROLE TO USER ELT_RL_RUNNER;

SELECT CURRENT_ORGANIZATION_NAME() || '-' || CURRENT_ACCOUNT_NAME() AS CONNECTOR_ACCOUNT,
       'ELT_RL_DB' AS EXPERIMENT_DATABASE,
       'ELT_RL_WH' AS EXPERIMENT_WAREHOUSE,
       'ELT_RL_RUNNER' AS EXPERIMENT_USER;
"""
    _write_private(bootstrap_path, script.encode())
    return {"bootstrap": bootstrap_path, "config": connection_path, "private_key": key_path}


def prepare_manager(base_config: dict, root: Path) -> dict[str, Path]:
    """Prepare a host-only resource manager; model jobs receive episode roles."""
    from elt_rl.task import identifier

    directory = Path(root) / ".snowflake"
    if directory.is_symlink():
        raise ValueError("Manager directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    key_path = directory / "native_manager_key.p8"
    if key_path.is_symlink():
        raise ValueError("Manager key cannot be a symlink")
    public_key = _public_key(key_path)
    database = identifier(base_config["database"]).upper()
    warehouse = identifier(base_config["warehouse"]).upper()
    config = {
        **base_config,
        "user": "ELT_RL_RESOURCE_MANAGER",
        "role": "ELT_RL_RESOURCE_MANAGER",
        "private_key_file": str(key_path.resolve()),
    }
    config.pop("password", None)
    config["authenticator"] = "SNOWFLAKE_JWT"
    config_path = directory / "native_manager.json"
    _write_private(config_path, (json.dumps(config, indent=2) + "\n").encode())
    sql = f"""USE ROLE ACCOUNTADMIN;
CREATE ROLE IF NOT EXISTS ELT_RL_RESOURCE_MANAGER;
GRANT ROLE ELT_RL_RESOURCE_MANAGER TO ROLE SYSADMIN;
GRANT CREATE ROLE ON ACCOUNT TO ROLE ELT_RL_RESOURCE_MANAGER;
GRANT USAGE ON DATABASE {database} TO ROLE ELT_RL_RESOURCE_MANAGER WITH GRANT OPTION;
GRANT CREATE SCHEMA ON DATABASE {database} TO ROLE ELT_RL_RESOURCE_MANAGER;
GRANT USAGE ON WAREHOUSE {warehouse} TO ROLE ELT_RL_RESOURCE_MANAGER WITH GRANT OPTION;
CREATE USER IF NOT EXISTS ELT_RL_RESOURCE_MANAGER TYPE=SERVICE DEFAULT_ROLE=ELT_RL_RESOURCE_MANAGER;
ALTER USER ELT_RL_RESOURCE_MANAGER SET RSA_PUBLIC_KEY='{public_key}';
GRANT ROLE ELT_RL_RESOURCE_MANAGER TO USER ELT_RL_RESOURCE_MANAGER;
ALTER USER ELT_RL_RESOURCE_MANAGER SET DEFAULT_SECONDARY_ROLES = ();
ALTER USER ELT_RL_RUNNER SET DEFAULT_SECONDARY_ROLES = ();
"""
    script = directory / "native_manager.sql"
    _write_private(script, sql.encode())
    return {"config": config_path, "bootstrap": script, "key": key_path}
