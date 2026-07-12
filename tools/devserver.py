#!/usr/bin/env python3
"""Dev/debug reference server for the getting-air WebGPU sim (PR-1 scope).

Stdlib-only (no pip install). Serves the working tree with no-cache so every edit
shows on a plain refresh, and accepts snapshot uploads on POST /collect. This is a
DEV TOOL: run it locally to iterate; GitHub Pages never executes it.

Scope note: this is the PR-1 server (static + /collect). The PR-2 short-poll control
loop (/hello, /cmd, /ack) is intentionally NOT here -- see
session dev-harness-design.md, which converged on adding it only if a demonstrated
cross-engine workflow needs it.

Security posture (see design doc section 7):
- Binds loopback (127.0.0.1) by default; --host 0.0.0.0 / --lan is an explicit opt-in.
- A per-run random token gates /collect; it is MANDATORY when bound non-loopback and
  kept on for loopback too. Pass it in the `X-Devharness-Token` header.
- Origin is a supplementary check: a request whose Origin is present AND cross-origin
  is rejected (a hostile web page reaching a loopback server); a same-origin request
  (or one with no Origin, e.g. curl) passes the Origin gate but still needs the token.
- /collect enforces a body-size cap, safe atomic filenames, a bounded storage dir, and
  simple newest-N retention -- to bound disk/memory use.

Usage:
  python3 tools/devserver.py                      # http, loopback, :8080
  python3 tools/devserver.py --port 4444          # (what chad's validate-all.js expects)
  python3 tools/devserver.py --https --port 4444  # self-signed TLS (cert auto-generated)
  python3 tools/devserver.py --lan                # bind 0.0.0.0 (token mandatory)
  python3 tools/devserver.py --token MYTOKEN      # fixed token instead of random
"""
import argparse
import http.server
import os
import re
import secrets
import ssl
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
COLLECT_DIR = os.path.join(ROOT, ".devharness-collect")            # git-ignored (see below)
MAX_UPLOAD = 64 * 1024 * 1024      # 64 MiB per snapshot (raw debugSnapshotSave can be big)
KEEP_LAST = 200                    # newest-N retention in COLLECT_DIR
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


class Handler(http.server.SimpleHTTPRequestHandler):
    # injected by main(): token, require_token
    token = ""
    require_token = True

    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    # --- no-cache on everything ---
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[devserver] %s %s\n" % (self.address_string(), fmt % args))

    # --- auth helpers ---
    def _origin_ok(self) -> bool:
        # Supplementary check: reject only a PRESENT, cross-origin Origin. "Same-origin"
        # is decided by comparing the Origin's host to the request's Host header, so it
        # stays correct regardless of the host/port the request actually arrived on (not
        # just the server's own bind address). A same-origin request or one with no
        # Origin (curl, some same-origin GETs) passes here; the token is the real gate.
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        host = self.headers.get("Host", "")
        return urllib.parse.urlparse(origin).netloc == host

    def do_GET(self):
        # Inject the dev-only activation marker (a per-run token) into served HTML, so
        # a page served BY THIS dev server can authenticate snapshot uploads and (PR-2)
        # enroll for control. A bare GitHub Pages deploy serves the committed HTML
        # WITHOUT this injection -> no marker -> the control client stays inert on prod.
        p = urllib.parse.urlparse(self.path).path
        if p.endswith(".html") or p.endswith("/"):
            fs = self.translate_path(self.path)
            if os.path.isdir(fs):
                fs = os.path.join(fs, "index.html")
            if os.path.isfile(fs):
                try:
                    with open(fs, "rb") as fh:
                        html = fh.read()
                    marker = ('<meta name="devharness-token" content="%s">' % self.token).encode()
                    m = re.search(rb"<head[^>]*>", html, re.IGNORECASE)
                    html = (html[:m.end()] + marker + html[m.end():]) if m else (marker + html)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                except OSError:
                    pass
        return super().do_GET()

    def _token_ok(self) -> bool:
        if not self.require_token:
            return True
        supplied = self.headers.get("X-Devharness-Token", "")
        return bool(self.token) and secrets.compare_digest(supplied, self.token)

    def _reject(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/collect":
            self._reject(404, b"not found")
            return
        if not self._origin_ok():
            self._reject(403, b"cross-origin rejected")
            return
        if not self._token_ok():
            self._reject(401, b"missing/invalid token")
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0 or length > MAX_UPLOAD:
            self._reject(413, b"empty or too large")
            return
        body = self.rfile.read(length)
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
        tag = _SAFE.sub("_", q.get("tag", "snap"))[:64] or "snap"
        step = _SAFE.sub("_", q.get("step", "0"))[:32] or "0"
        ext = _SAFE.sub("", q.get("ext", "bin"))[:8] or "bin"
        os.makedirs(COLLECT_DIR, exist_ok=True)
        base = "%s-%s" % (tag, step)
        # atomic write: tmp then rename within the same dir
        fd, tmp = None, os.path.join(COLLECT_DIR, base + ".%s.tmp" % secrets.token_hex(4))
        with open(tmp, "wb") as fh:
            fh.write(body)
        final = os.path.join(COLLECT_DIR, base + "." + ext)
        os.replace(tmp, final)
        self._retain()
        sys.stderr.write("[devserver] POST %dB -> %s\n" % (len(body), os.path.relpath(final, ROOT)))
        self._reject(200, b"ok")

    def _retain(self):
        try:
            files = [os.path.join(COLLECT_DIR, f) for f in os.listdir(COLLECT_DIR)]
            files = [f for f in files if os.path.isfile(f) and not f.endswith(".tmp")]
            files.sort(key=os.path.getmtime)
            for f in files[:-KEEP_LAST]:
                os.remove(f)
        except OSError:
            pass


def _self_signed_cert():
    """Generate an ephemeral self-signed cert in COLLECT_DIR (stdlib ssl can't mint one;
    needs the `cryptography` pkg or openssl). We shell out to openssl if present; else
    tell the user to use --http (loopback http is already a WebGPU secure context).
    Includes SANs for localhost/127.0.0.1/::1 and this host's primary IP, so the cert
    matches when reached by LAN IP (the browser still warns it's self-signed -- accept
    once -- but it won't ALSO be a host-mismatch, which some browsers block harder)."""
    import subprocess
    import socket
    os.makedirs(COLLECT_DIR, exist_ok=True)
    cert = os.path.join(COLLECT_DIR, "devcert.pem")
    key = os.path.join(COLLECT_DIR, "devkey.pem")
    # best-effort detect the primary outbound IP
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    san = "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1,IP:%s" % ip
    if not (os.path.exists(cert) and os.path.exists(key)):
        try:
            subprocess.run(
                ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                 "-keyout", key, "-out", cert, "-days", "365",
                 "-subj", "/CN=localhost", "-addext", san],
                check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            sys.exit("--https needs openssl to mint a self-signed cert; "
                     "or just use --http (loopback http is a WebGPU secure context).")
    return cert, key


def main():
    ap = argparse.ArgumentParser(description="getting-air dev/debug server (static + /collect)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    ap.add_argument("--lan", action="store_true", help="bind 0.0.0.0 (token mandatory)")
    ap.add_argument("--https", action="store_true", help="serve TLS (self-signed)")
    ap.add_argument("--token", default=None, help="fixed token (default: per-run random)")
    args = ap.parse_args()

    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    loopback = _is_loopback(host)
    token = args.token or secrets.token_urlsafe(16)

    Handler.token = token
    Handler.require_token = True  # on even for loopback; mandatory when non-loopback
    scheme = "https" if args.https else "http"

    httpd = http.server.ThreadingHTTPServer((host, args.port), Handler)
    if args.https:
        cert, key = _self_signed_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert, keyfile=key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    if not loopback:
        sys.stderr.write("[devserver] WARNING: bound non-loopback %s -- token REQUIRED, "
                         "and a LAN IP over http is not a WebGPU secure context "
                         "(use --https or the browser insecure-origin flag).\n" % host)
    url = "%s://%s:%d/" % (scheme, "localhost" if loopback else host, args.port)
    sys.stderr.write("[devserver] serving %s at %s (no-cache)\n" % (ROOT, url))
    sys.stderr.write("[devserver] /collect token (X-Devharness-Token): %s\n" % token)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
