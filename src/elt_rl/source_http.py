"""Read-only source HTTP process. The model cannot alter its route map or files."""

import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROUTES = json.loads(Path("routes.json").read_text())


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = ROUTES.get(urlsplit(self.path).path)
        if route is None:
            self.send_error(404)
            return
        content = Path(route["file"]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", route["type"])
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if Path("server.crt").exists():
        secure = HTTPServer(("0.0.0.0", 8443), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain("server.crt", "server.key")
        secure.socket = context.wrap_socket(secure.socket, server_side=True)
        threading.Thread(target=secure.serve_forever, daemon=True).start()
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
