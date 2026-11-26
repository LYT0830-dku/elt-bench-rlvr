"""Real loopback source services owned by an ordinary user, without containers.

PostgreSQL preserves its original type system; HTTP/TLS serve the original data;
Moto implements the S3 wire API, as an explicit replacement for LocalStack.
Actors only receive logical capabilities, never service credentials or addresses.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from elt_rl.artifacts import write_json
from elt_rl.errors import InfrastructureError


def loopback_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def create_tls(root):
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ELT episode loopback")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = root / "source.crt", root / "source.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return cert_path, key_path


class NativeSourceServices:
    def __init__(self, task, rows, root, config, deadline=None):
        self.task, self.rows, self.root = task, rows, Path(root).resolve()
        self.root.mkdir(parents=True, mode=0o700)
        self.config = config
        self.deadline = deadline or float("inf")
        self.http_token = secrets.token_urlsafe(32)
        self.servers, self.threads, self.processes = [], [], []
        self.streams = {}
        self.events = []
        self.s3 = None
        self.closed = False
        self.record("initializing")
        try:
            from elt_rl.source_adapters import SOURCES

            SOURCES.initialize(self)
            self.record("ready")
        except BaseException:
            self.close()
            raise

    def record(self, state):
        write_json(
            self.root / "resources.json",
            {
                "backend": "native_services_v1",
                "state": state,
                "processes": [{"pid": p.pid, "running": p.poll() is None} for p in self.processes],
                "listeners": [{"port": s.server_port, "host": "127.0.0.1"} for s in self.servers],
                "events": self.events,
            },
        )

    def _serve(self, server):
        self.servers.append(server)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.threads.append(thread)
        thread.start()

    def _http(self):
        routes = {}
        for source in self.task.sources:
            if source.kind not in {"file", "http", "https", "csv_http", "rest"}:
                continue
            route = source.options.get("url_path", "/" + source.name)
            if source.kind == "rest":
                import numpy as np
                import pandas as pd

                # The original REST server uses read_csv -> NaN/None -> records.
                # Preserve the original header case and pandas numeric/null semantics.
                frame = pd.read_csv(source.options["path"]).replace({np.nan: None})
                content, fmt = (
                    json.dumps(frame.to_dict(orient="records"), allow_nan=False).encode(),
                    "json",
                )
            else:
                content = Path(source.options["path"]).read_bytes()
                fmt = source.options.get("format", "csv")
            routes[route] = (content, fmt)
            self.streams[source.name] = {
                "kind": "https" if source.kind in {"csv_http", "https"} else "http",
                "path": route,
                "format": fmt,
            }
        token = self.http_token

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    self.send_error(403)
                    return
                item = routes.get(self.path)
                if item is None:
                    self.send_error(404)
                    return
                body, fmt = item
                self.send_response(200)
                self.send_header(
                    "Content-Type", "application/json" if fmt in {"json", "jsonl"} else "text/csv"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._serve(http)
        certificate, key = create_tls(self.root)
        secure = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        secure.socket = context.wrap_socket(secure.socket, server_side=True)
        self._serve(secure)
        for value in self.streams.values():
            tls = value["kind"] == "https"
            value.update(
                url=f"{'https' if tls else 'http'}://127.0.0.1:{secure.server_port if tls else http.server_port}{value['path']}",
                ca=str(certificate) if tls else None,
            )

    def _postgres(self):
        import psycopg
        from psycopg import sql

        configured = self.config.get("postgres_bin")
        if configured:
            bindir = Path(configured).resolve()
        else:
            binary = shutil.which("pg_config")
            bindir = (
                Path(subprocess.check_output([binary, "--bindir"], text=True).strip())
                if binary
                else None
            )
        if (
            bindir is None
            or not (bindir / "postgres").is_file()
            or not (bindir / "initdb").is_file()
        ):
            raise InfrastructureError(
                "Configure an existing user-runnable PostgreSQL binary directory"
            )
        data = self.root / "postgres"
        password = secrets.token_urlsafe(32)
        pw = self.root / "pg-init-password"
        pw.write_text(password)
        pw.chmod(0o600)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "LC_ALL": "C"}
        try:
            result = subprocess.run(
                [
                    str(bindir / "initdb"),
                    "-D",
                    str(data),
                    "-U",
                    "postgres",
                    "--pwfile",
                    str(pw),
                    "--auth",
                    "scram-sha-256",
                    "--encoding",
                    "UTF8",
                    "--no-locale",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        finally:
            pw.unlink(missing_ok=True)
        (self.root / "pg-init.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise InfrastructureError(
                "User-space PostgreSQL initialization failed; inspect private source log"
            )
        port = loopback_port()
        log = (self.root / "postgres.log").open("w")
        process = subprocess.Popen(
            [
                str(bindir / "postgres"),
                "-D",
                str(data),
                "-p",
                str(port),
                "-c",
                "listen_addresses=127.0.0.1",
                "-c",
                "unix_socket_directories=",
                "-c",
                "shared_buffers=16MB",
                "-c",
                "max_connections=12",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        log.close()
        self.processes.append(process)
        self.record("postgres_starting")
        until = min(self.deadline, time.monotonic() + 45)
        connection = None
        while time.monotonic() < until and process.poll() is None:
            try:
                connection = psycopg.connect(
                    host="127.0.0.1",
                    port=port,
                    user="postgres",
                    password=password,
                    dbname="postgres",
                    connect_timeout=2,
                )
                break
            except psycopg.OperationalError:
                time.sleep(0.2)
        if connection is None:
            raise InfrastructureError("User-space PostgreSQL did not become ready")
        reader = secrets.token_urlsafe(32)
        connection.autocommit = True
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.task.name)))
        connection.close()
        connection = psycopg.connect(
            host="127.0.0.1",
            port=port,
            user="postgres",
            password=password,
            dbname=self.task.name,
            connect_timeout=10,
        )
        with connection:
            for source in self.task.sources:
                if source.kind != "postgres":
                    continue
                definitions = source.options["initial_types"]
                if set(definitions) != set(source.columns) or any(
                    t
                    not in {
                        "REAL",
                        "SERIAL",
                        "INTEGER",
                        "BIGINT",
                        "TEXT",
                        "DOUBLE PRECISION",
                        "VARCHAR",
                        "DATE",
                        "TIMESTAMP",
                        "BOOLEAN",
                    }
                    for t in definitions.values()
                ):
                    raise InfrastructureError(
                        "Unrecognized original PostgreSQL initialization schema"
                    )
                connection.execute(
                    sql.SQL("CREATE TABLE {} ({})").format(
                        sql.Identifier(source.name),
                        sql.SQL(",").join(
                            sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(definitions[c]))
                            for c in source.columns
                        ),
                    )
                )
                keys = source.options.get("initial_primary_keys", [])
                if keys:
                    connection.execute(
                        sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(
                            sql.Identifier(source.name),
                            sql.SQL(",").join(map(sql.Identifier, keys)),
                        )
                    )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        sql.SQL("INSERT INTO {} VALUES ({})").format(
                            sql.Identifier(source.name),
                            sql.SQL(",").join(sql.Placeholder() for _ in source.columns),
                        ),
                        [tuple(row[c] for c in source.columns) for row in self.rows[source.name]],
                    )
                self.streams[source.name] = {
                    "kind": "postgres",
                    "table": source.name,
                    "format": "table",
                }
            connection.execute(
                sql.SQL("CREATE ROLE source_reader LOGIN PASSWORD {}").format(sql.Literal(reader))
            )
            connection.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO source_reader").format(
                    sql.Identifier(self.task.name)
                )
            )
            connection.execute("GRANT USAGE ON SCHEMA public TO source_reader")
            connection.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO source_reader")
        self.pg_config = dict(
            host="127.0.0.1",
            port=port,
            user="source_reader",
            password=reader,
            dbname=self.task.name,
            connect_timeout=10,
        )

    def _s3(self):
        import boto3
        from botocore.auth import S3SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.config import Config
        from botocore.credentials import Credentials
        from moto.server import create_backend_app
        from werkzeug.serving import WSGIRequestHandler, make_server
        from werkzeug.wrappers import Request

        access, secret = secrets.token_hex(12), secrets.token_urlsafe(32)
        self.bucket = "elt-" + secrets.token_hex(16)
        backend = create_backend_app("s3")
        self.s3_readonly = False

        def application(environ, respond):
            request = Request(environ)
            body = request.get_data()
            valid = False
            try:
                auth = request.headers.get("Authorization", "")
                prefix, fields = auth.split(" ", 1)
                parts = dict(part.strip().split("=", 1) for part in fields.split(","))
                credential = parts["Credential"].split("/")
                signed = parts["SignedHeaders"].split(";")
                timestamp = request.headers["X-Amz-Date"]
                age = abs(
                    (
                        datetime.now(timezone.utc)
                        - datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(
                            tzinfo=timezone.utc
                        )
                    ).total_seconds()
                )
                headers = {name: request.headers[name] for name in signed}
                raw_uri = environ.get("RAW_URI", request.full_path.rstrip("?"))
                wire = AWSRequest(
                    method=request.method,
                    url="http://" + request.host + raw_uri,
                    headers=headers,
                    data=body,
                )
                wire.context["timestamp"] = timestamp
                signer = S3SigV4Auth(Credentials(access, secret), "s3", "us-west-2")
                signature = signer.signature(
                    signer.string_to_sign(wire, signer.canonical_request(wire)), wire
                )
                valid = (
                    prefix == "AWS4-HMAC-SHA256"
                    and credential == [access, timestamp[:8], "us-west-2", "s3", "aws4_request"]
                    and age < 900
                    and hmac.compare_digest(signature, parts["Signature"])
                    and (
                        request.path == "/" + self.bucket
                        or request.path.startswith("/" + self.bucket + "/")
                    )
                    and (not self.s3_readonly or request.method in {"GET", "HEAD"})
                )
            except (KeyError, ValueError):
                pass
            if not valid:
                respond("403 Forbidden", [("Content-Type", "application/xml")])
                return [b"<Error><Code>AccessDenied</Code></Error>"]
            environ["wsgi.input"] = io.BytesIO(body)
            return backend(environ, respond)

        class Quiet(WSGIRequestHandler):
            def log(self, *args, **kwargs):
                pass

        server = make_server("127.0.0.1", 0, application, threaded=True, request_handler=Quiet)
        self._serve(server)
        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"http://127.0.0.1:{server.server_port}",
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="us-west-2",
            config=Config(retries={"max_attempts": 0}, connect_timeout=10, read_timeout=20),
        )
        self.s3.create_bucket(
            Bucket=self.bucket, CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
        for source in self.task.sources:
            if source.kind == "s3":
                key = source.options["key"]
                self.s3.put_object(
                    Bucket=self.bucket, Key=key, Body=Path(source.options["path"]).read_bytes()
                )
                self.streams[source.name] = {
                    "kind": "s3",
                    "key": key,
                    "format": source.options.get("format", "jsonl"),
                }
        self.s3_readonly = True

    def public(self):
        return {
            name: {
                "source_id": name,
                "connector": item["kind"],
                "format": item["format"],
                "columns": next(s.columns for s in self.task.sources if s.name == name),
            }
            for name, item in self.streams.items()
        }

    def extract(self, stream, connector, fmt, columns):
        from elt_rl.source_adapters import extract

        return extract(self, stream, connector, fmt, columns)

    def close(self):
        if self.closed:
            return
        failures = []
        if self.s3:
            try:
                self.s3_readonly = False
                contents = self.s3.list_objects_v2(Bucket=self.bucket).get("Contents", [])
                for item in contents:
                    self.s3.delete_object(Bucket=self.bucket, Key=item["Key"])
                self.s3.delete_bucket(Bucket=self.bucket)
            except Exception as exc:
                failures.append(type(exc).__name__)
            finally:
                self.s3.close()
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=10)
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        self.servers = []
        self.closed = not failures
        self.record("closed" if not failures else "cleanup_failed")
        if failures:
            raise InfrastructureError("Owned source cleanup failed: " + str(failures))
