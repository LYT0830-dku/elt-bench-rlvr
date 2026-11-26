"""Actual HTTP/TLS/S3 requests without Docker, daemon, cloud credentials or model calls."""

import builtins
import json
import ssl
from dataclasses import replace

import httpx
import pytest

from elt_rl.native_sources import NativeSourceServices
from elt_rl.source_data import source_rows
from elt_rl.task import Source


def test_authenticated_https_original_bytes_and_cleanup(task, tmp_path):
    path = tmp_path / "original.csv"
    original = b"id,name\n001,Alice\n002,Bob\n"
    path.write_bytes(original)
    variant = replace(
        task,
        sources=(
            Source(
                "input",
                "csv_http",
                ("id", "name"),
                {"path": str(path), "format": "csv", "url_path": "/input.csv"},
            ),
        ),
    )
    services = NativeSourceServices(variant, source_rows(variant), tmp_path / "services", {})
    endpoint = services.streams["input"]["url"]
    cert = services.streams["input"]["ca"]
    with httpx.Client(trust_env=False, verify=ssl.create_default_context(cafile=cert)) as client:
        assert client.get(endpoint).status_code == 403
        result = client.get(endpoint, headers={"Authorization": "Bearer " + services.http_token})
        assert result.content == original
        assert (
            client.get(
                endpoint + "/../source.key",
                headers={"Authorization": "Bearer " + services.http_token},
            ).status_code
            == 404
        )
    assert services.extract("input", "https", "csv", ("id", "name"))[0] == {
        "id": "001",
        "name": "Alice",
    }
    assert services.http_token not in json.dumps(services.public())
    services.close()
    assert json.loads((tmp_path / "services/resources.json").read_text())["state"] == "closed"
    with pytest.raises(httpx.ConnectError):
        httpx.get(
            endpoint, verify=ssl.create_default_context(cafile=cert), trust_env=False, timeout=1
        )


def test_s3_signed_requests_read_only_scope_and_no_docker_import(task, tmp_path, monkeypatch):
    path = tmp_path / "country.jsonl"
    path.write_text('{"id":1,"name":"Japan"}\n')
    variant = replace(
        task,
        sources=(
            Source(
                "country",
                "s3",
                ("id", "name"),
                {"path": str(path), "format": "jsonl", "key": "country.jsonl"},
            ),
        ),
    )
    original_import = builtins.__import__

    def no_container(name, *args, **kwargs):
        if name == "docker" or name.startswith("docker."):
            pytest.fail("Native S3 attempted to import Docker")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_container)
    services = NativeSourceServices(variant, source_rows(variant), tmp_path / "s3", {})
    try:
        endpoint = services.s3.meta.endpoint_url
        assert (
            httpx.get(
                endpoint + "/" + services.bucket + "/country.jsonl", trust_env=False
            ).status_code
            == 403
        )
        assert services.extract("country", "s3", "jsonl", ("id", "name")) == [
            {"id": 1, "name": "Japan"}
        ]
        with pytest.raises(Exception, match="AccessDenied"):
            services.s3.put_object(Bucket=services.bucket, Key="evil", Body=b"bad")
        with pytest.raises(Exception, match="AccessDenied"):
            services.s3.get_object(Bucket="another-episode", Key="country.jsonl")
    finally:
        services.close()
