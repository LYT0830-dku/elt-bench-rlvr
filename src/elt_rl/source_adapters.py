"""Registered real source readers; only host-issued capabilities reach adapters."""

from __future__ import annotations

import csv
import io
import json
import ssl
from pathlib import Path

from elt_rl.errors import ActionError, InfrastructureError
from elt_rl.registry import Registry, UnsupportedCapability


class SourceAdapter:
    connector: str
    implemented = True
    initialization: str | None = None

    def initialize(self, services):
        if self.initialization:
            getattr(services, "_" + self.initialization)()

    def schema(self, source):
        return {
            "columns": source.columns,
            "configuration": {"format": source.options.get("format")},
        }

    def check_connection(self, services, stream):
        if services.closed or stream not in services.streams:
            raise InfrastructureError("Source session is closed or uninitialized")
        return {"connected": True, "connector": self.connector}

    def status(self, services):
        return {
            "closed": services.closed,
            "processes": [{"pid": p.pid, "running": p.poll() is None} for p in services.processes],
        }

    def load(self, backend, **arguments):
        return backend.load(**arguments)

    def cleanup(self, services):
        services.close()

    def read(self, services, info, fmt, columns):
        raise NotImplementedError


def decode(body, fmt, columns):
    if len(body) > 256 * 1024 * 1024:
        raise InfrastructureError("Source exceeds bounded extraction size")
    if fmt == "json":
        rows = json.loads(body)
    elif fmt == "jsonl":
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]
    elif fmt == "csv":
        rows = list(csv.DictReader(io.StringIO(body.decode())))
    else:
        raise UnsupportedCapability("Unsupported source format: " + fmt)
    folded = [{key.lower(): value for key, value in row.items()} for row in rows]
    return [{column: row[column] for column in columns} for row in folded]


class PostgreSQLAdapter(SourceAdapter):
    connector, initialization = "postgres", "postgres"

    def read(self, services, info, fmt, columns):
        import psycopg
        from psycopg import sql

        with psycopg.connect(**services.pg_config) as connection:
            query = sql.SQL("SELECT {} FROM public.{}").format(
                sql.SQL(",").join(map(sql.Identifier, columns)), sql.Identifier(info["table"])
            )
            # Preserve exact PostgreSQL binary float4 behavior from the baseline.
            return [
                dict(zip(columns, row, strict=True))
                for row in connection.execute(query, binary=True).fetchall()
            ]


class HTTPAdapter(SourceAdapter):
    connector, initialization = "http", "http"

    def read(self, services, info, fmt, columns):
        import httpx

        with httpx.Client(
            trust_env=False,
            verify=ssl.create_default_context(cafile=info["ca"]) if info["ca"] else True,
            follow_redirects=False,
            timeout=30,
        ) as client:
            response = client.get(
                info["url"], headers={"Authorization": "Bearer " + services.http_token}
            )
            response.raise_for_status()
            body = response.content
        return decode(body, fmt, columns)


class HTTPSAdapter(HTTPAdapter):
    connector = "https"


class S3Adapter(SourceAdapter):
    connector, initialization = "s3", "s3"

    def read(self, services, info, fmt, columns):
        body = services.s3.get_object(Bucket=services.bucket, Key=info["key"])["Body"].read()
        return decode(body, fmt, columns)


class LocalFileAdapter(SourceAdapter):
    connector, initialization = "local_file", None

    def initialize(self, services):
        for source in services.task.sources:
            if source.kind == "local_file":
                path = Path(source.options["path"])
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > 256 * 1024 * 1024
                ):
                    raise UnsupportedCapability("Unavailable/unsafe declared local source asset")
                services.streams[source.name] = {
                    "kind": self.connector,
                    "format": source.options.get("format", "csv"),
                    "local_path": str(path.resolve()),
                }

    def read(self, services, info, fmt, columns):
        return decode(Path(info["local_path"]).read_bytes(), fmt, columns)


class UnsupportedSourceAdapter(SourceAdapter):
    implemented = False

    def __init__(self, connector, reason):
        self.connector, self.reason = connector, reason

    def initialize(self, services):
        raise UnsupportedCapability(self.reason)


class SourceRegistry(Registry[SourceAdapter]):
    def __init__(self):
        super().__init__("source adapter")

    def validate(self, sources):
        for source in sources:
            adapter = self.get(source.kind)
            if not adapter.implemented:
                raise UnsupportedCapability(adapter.reason)

    def initialize(self, services):
        self.validate(services.task.sources)
        needed = {self.get(source.kind).initialization for source in services.task.sources}
        # The baseline's loopback HTTP/TLS pair and initialization ordering are
        # retained; connectors own every protocol-specific extraction operation.
        initialized = set()
        for name in ("http", "postgres", "s3"):
            if name == "http" or name in needed:
                self.get(name).initialize(services)
                initialized.add(name)
        for source in services.task.sources:
            adapter = self.get(source.kind)
            if adapter.initialization not in initialized:
                adapter.initialize(services)
                initialized.add(adapter.initialization)


SOURCES = SourceRegistry()
for _adapter in (
    PostgreSQLAdapter(),
    HTTPAdapter(),
    HTTPSAdapter(),
    S3Adapter(),
    LocalFileAdapter(),
):
    SOURCES.register(_adapter.connector, _adapter)
for _alias in ("rest", "file", "csv_http"):
    SOURCES.register(_alias, SOURCES.get("https" if _alias == "csv_http" else "http"))
SOURCES.register(
    "mongodb",
    UnsupportedSourceAdapter(
        "mongodb",
        "MongoDB provisioning/authorized server credentials are not implemented in this deployment; configure a verified adapter before execution",
    ),
)


def extract(services, stream, connector, fmt, columns):
    info = services.streams[stream]
    if connector != info["kind"] or fmt != info["format"]:
        raise ActionError("Connector/format does not match the selected source capability")
    return SOURCES.get(connector).read(services, info, fmt, columns)
