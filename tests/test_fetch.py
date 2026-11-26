import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from elt_rl.fetch import RangeReader


def test_range_zip_and_fail_closed_full_download():
    stream = io.BytesIO()
    payload = b"a" * 100_000
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("data/task/source.csv", payload)
    content = stream.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ignore-range":
                self.send_response(200)
                self.end_headers()
                return
            spec = self.headers["Range"].removeprefix("bytes=")
            left, right = spec.split("-")
            start = int(left) if left else max(0, len(content) - int(right))
            end = int(right) if left and right else len(content) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.end_headers()
            self.wfile.write(content[start : end + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with RangeReader(base + "/archive") as reader, zipfile.ZipFile(reader) as archive:
            assert archive.read("data/task/source.csv") == payload
        with pytest.raises(ValueError, match="refusing full archive"):
            RangeReader(base + "/ignore-range")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
