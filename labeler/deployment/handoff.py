"""Temporary localhost handoff: old tabs write to 101, never to the retired database."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlsplit
from pathlib import Path


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, headers):
        self.send_response(code)
        for name in ("Content-Type", "Content-Disposition", "X-Content-Type-Options"):
            if headers.get(name):
                self.send_header(name, headers[name])
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self):
        try:
            expected = "127.0.0.1:" + str(self.server.server_port)
            if self.headers.get("Host") != expected or self.headers.get("Origin", "http://" + expected) != "http://" + expected:
                raise PermissionError("请从原分类页面操作")
            if not self.path.startswith("/api/"):
                self.send_response(302)
                self.send_header("Location", self.server.target + "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            headers, body = {}, None
            if self.command == "POST":
                if self.headers.get("X-Local-Token") not in self.server.tokens:
                    raise PermissionError("请刷新分类页面")
                length = int(self.headers.get("Content-Length", "0"))
                if not 2 <= length <= 2 * 1024 * 1024:
                    raise ValueError("请求格式不正确")
                body = self.rfile.read(length)
                headers = {"Content-Type": "application/json", "X-Local-Token": self.server.remote_token}
            request = Request(self.server.target + self.path, data=body, headers=headers, method=self.command)
            try:
                response = urlopen(request, timeout=65)
            except HTTPError as exc:
                response = exc
            with response:
                data = response.read()
                if self.path == "/api/state" and response.status == 200:
                    token = json.loads(data)["token"]
                    self.server.remote_token = token
                    self.server.tokens.add(token)
                self.reply(response.status, data, response.headers)
        except (BrokenPipeError, ConnectionError):
            pass
        except Exception as exc:
            data = json.dumps({"error": str(exc)}, ensure_ascii=False).encode()
            self.reply(403 if isinstance(exc, PermissionError) else 502, data,
                       {"Content-Type": "application/json; charset=utf-8"})

    do_GET = forward
    do_POST = forward


def make_proxy(target, old_token, port=8878):
    if urlsplit(target).scheme != "http":
        raise ValueError("Expected the internal HTTP service")
    with urlopen(target.rstrip("/") + "/api/state", timeout=10) as response:
        remote = json.load(response)
    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    server.daemon_threads = True
    server.target = target.rstrip("/")
    server.remote_token = remote["token"]
    server.tokens = {old_token, remote["token"]}
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    make_proxy(config["url"], config["old_token"]).serve_forever()
