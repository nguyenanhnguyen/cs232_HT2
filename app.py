"""CS232 HT2 - Web app (chỉ dùng thư viện chuẩn).  Chạy: python app.py  -> http://localhost:8000"""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import core

PORT = 8000
index = core.Index()


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send((core.ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/images/"):
            f = core.IMG_DIR / self.path.split("/")[-1]
            if f.name in index.files:
                return self._send(f.read_bytes(), "image/" + f.suffix.lstrip(".").replace("jpg", "jpeg"))
        self._send(b"Not found", "text/plain", 404)

    def do_POST(self):  # /search  body: {"image": "<base64>"}
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = base64.b64decode(req["image"].split(",")[-1])
        res = {m: [{"id": i, "file": f, "score": round(s, 4)} for i, f, s in index.search(data, m)]
               for m in core.METHODS}
        self._send(json.dumps({"names": core.METHODS, "results": res}).encode(), "application/json")


if __name__ == "__main__":
    print(f"http://localhost:{PORT}")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
