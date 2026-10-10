#!/usr/bin/env python3
"""Local preview server for the MultINav project page.

Serves the folder containing this script over HTTP with byte-range support, so
the browser can seek inside the MP4 files in static/videos.

    python serve.py            # http://127.0.0.1:8811
    python serve.py 9000       # custom port
"""

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with single-range support for media files."""

    def send_head(self):
        self._range = None
        header = self.headers.get("Range")
        if not header:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        match = RANGE_RE.match(header.strip())
        if not match:
            return super().send_head()

        size = os.path.getsize(path)
        start_raw, end_raw = match.groups()
        if start_raw == "":
            length = int(end_raw or 0)
            if length <= 0:
                self.send_error(416, "Invalid range")
                return None
            start = max(size - length, 0)
            end = size - 1
        else:
            start = int(start_raw)
            end = int(end_raw) if end_raw else size - 1
            end = min(end, size - 1)

        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return None

        handle = open(path, "rb")
        handle.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._range = (start, end)
        self._source = handle
        return handle

    def copyfile(self, source, outputfile):
        if getattr(self, "_range", None) is None:
            return super().copyfile(source, outputfile)
        start, end = self._range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The player closed the connection, which is normal for seeks.
                break
            remaining -= len(chunk)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def end_headers(self):
        if "Accept-Ranges" not in self._headers_buffer_text():
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def _headers_buffer_text(self):
        try:
            return b"".join(self._headers_buffer).decode("latin-1")
        except AttributeError:
            return ""

    def log_message(self, fmt, *args):
        message = fmt % args
        if " 206 " in message or " 200 " in message:
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), message))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8811
    handler = partial(RangeRequestHandler, directory=ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print("MultINav project page: http://127.0.0.1:%d/" % port)
    print("Serving %s (Range supported, Ctrl+C to stop)" % ROOT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
