#!/usr/bin/env python3

"""Simple https server for development."""

import datetime
import json
import os
import ssl
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

# The cert lives OUTSIDE the served directory on purpose. This handler serves
# the current working directory verbatim, so a key kept next to index.html is
# downloadable by anyone who can reach the server -- which matters now that
# this binds every interface (see server_address below), not just loopback.
# Falls back to the old in-tree cert so an existing checkout still runs.
_LAN_CERT = os.path.expanduser('~/.config/getting-air/lan.pem')
CERTFILE = _LAN_CERT if os.path.exists(_LAN_CERT) else './localhost.pem'


def main():
    https_server(certfile=CERTFILE)


# ── Telemetry back channel ───────────────────────────────────────────────────
# Devices that are not this machine (a phone on the LAN, say) have no CDP
# endpoint to attach to, so their performance is otherwise invisible. Pages
# loaded with ?telemetry=1 POST periodic samples here and they are appended as
# newline-delimited JSON to TELEMETRY_LOG, which can then be read like any
# other local file.
#
# Deliberately opt-in per page load, same-origin only, and local-only: nothing
# is forwarded anywhere, the log is a plain file in the repo directory (which
# .gitignore already excludes via *.log), and with no ?telemetry=1 the page
# never posts at all.
TELEMETRY_LOG = './telemetry.log'
_telemetry_lock = threading.Lock()


class DevRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/_telemetry':
            self.send_error(404, 'not found')
            return
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > 64 * 1024:
                self.send_error(413, 'bad length')
                return
            body = self.rfile.read(n)
            rec = json.loads(body.decode('utf-8'))
        except Exception as e:
            self.send_error(400, f'bad payload: {e}')
            return
        rec['_server_time'] = datetime.datetime.now().isoformat(timespec='seconds')
        rec['_peer'] = self.client_address[0]
        with _telemetry_lock:
            with open(TELEMETRY_LOG, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    # NEVER LET THE BROWSER REUSE A FILE. With only Last-Modified and no
    # Cache-Control, Chrome may heuristically serve a cached main*.js or
    # shaders/*.wgsl without asking -- so an edit is not what runs, and a CDP
    # A/B of two override values compares the OLD build against itself. That
    # happened (plans/2D-backport.md B6-8): two render modes came out
    # byte-identical until the cache was disabled. A dev server is the one
    # place where every load must be the file on disk.
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):
        # Telemetry POSTs would otherwise flood the console every few seconds.
        if 'POST /_telemetry' in (fmt % args):
            return
        super().log_message(fmt, *args)


def https_server(*, certfile):
    print(f'`https_server()` starts... certfile={certfile}')
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile)

    # '' binds every interface, so this is reachable from the LAN, not just
    # localhost. WebGPU needs a secure context; https:// with a self-signed
    # cert qualifies once the browser has been told to trust it (localhost is
    # exempt and needs no click-through).
    server_address = ('', int(os.environ.get('GA_PORT', 4444)))
    with HTTPServer(server_address, DevRequestHandler) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        print_server_info(httpd)
        try:
            httpd.serve_forever()
        except Exception as e:
            httpd.server_close()
            raise e


def print_server_info(server):
    print(f"""Server info:
    name: {server.server_name}
    address: {server.server_address}
    """)


if __name__ == "__main__":
    main()
