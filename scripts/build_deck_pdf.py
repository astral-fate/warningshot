"""Render the pitch deck to PDF: one slide per 16:9 page.

The deck is the primary format -- it is interactive and it is what GitHub Pages
serves. But Pages on a private repository needs a paid plan, and a PDF is what
survives being emailed, attached to a submission, or opened by someone who will
not clone anything. This builds that copy from the same index.html, so the two
cannot drift.

The page is served over HTTP rather than opened as a file:// URL, because the
deck loads docs/pipeline.svg as a relative asset and some renderers refuse
subresources on file://. Backgrounds are printed explicitly; the deck is a dark
design and a PDF of it with backgrounds dropped is pale grey text on white.

    python scripts/build_deck_pdf.py [--out docs/deck.pdf] [--open]

Requires Playwright and a Chromium-family browser. It prefers an already
installed Chrome or Edge over downloading one:

    pip install playwright
"""

import argparse
import contextlib
import functools
import http.server
import socket
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "deck.pdf"

# 16:9 at the size PowerPoint and Keynote use, so the PDF drops into a slot
# built for slides without rescaling.
PAGE_W_IN = 13.333
PAGE_H_IN = 7.5


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory):
    """Serve `directory` on localhost for the lifetime of the block."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

        def handle_error(self, request, client_address):  # noqa: D102
            pass

    port = _free_port()
    httpd = Quiet(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d/" % port
    finally:
        httpd.shutdown()
        httpd.server_close()


def _launch(pw):
    """Prefer an installed browser; fall back to Playwright's own."""
    errors = []
    for channel in ("chrome", "msedge"):
        try:
            return pw.chromium.launch(channel=channel), channel
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (channel, str(exc).splitlines()[0]))
    try:
        return pw.chromium.launch(), "bundled chromium"
    except Exception as exc:  # noqa: BLE001
        errors.append("bundled: %s" % str(exc).splitlines()[0])
    raise SystemExit(
        "no Chromium-family browser available.\n  " + "\n  ".join(errors)
        + "\n\nInstall one, or run:  python -m playwright install chromium"
    )


def build(out_path, deck="index.html"):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("playwright is not installed.  pip install playwright")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with serve(ROOT) as base, sync_playwright() as pw:
        browser, which = _launch(pw)
        print("    rendering with %s" % which)
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        failures = []
        page.on("requestfailed",
                lambda r: failures.append("%s (%s)" % (r.url, r.failure)))
        page.on("pageerror", lambda e: failures.append("JS error: %s" % e))

        page.goto(base + deck, wait_until="networkidle")
        # The diagram is the one subresource; a silent 404 would print a blank
        # panel where the methodology slide's figure should be.
        page.wait_for_selector("img[src$='pipeline.svg']", state="attached")
        ok = page.evaluate(
            "() => { const i = document.querySelector(\"img[src$='pipeline.svg']\");"
            "  return !!(i && i.complete && i.naturalWidth > 0); }"
        )
        if not ok:
            raise SystemExit("the pipeline diagram did not load; refusing to "
                             "write a PDF with a blank figure")

        n_slides = page.evaluate("() => document.querySelectorAll('.slide').length")

        page.emulate_media(media="print")
        page.pdf(path=str(out_path),
                 width="%sin" % PAGE_W_IN, height="%sin" % PAGE_H_IN,
                 print_background=True, prefer_css_page_size=True,
                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        browser.close()

    if failures:
        print("    warning: %d resource/JS problem(s) while rendering:" % len(failures))
        for f in failures[:5]:
            print("      " + f)

    return out_path, n_slides


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output path (default: docs/deck.pdf)")
    ap.add_argument("--open", action="store_true", dest="open_after",
                    help="open the PDF when it is written")
    args = ap.parse_args()

    print("\n[deck] rendering index.html to PDF")
    out, n_slides = build(args.out)

    size_kb = out.stat().st_size / 1024.0
    pages = None
    try:
        import pypdf
        pages = len(pypdf.PdfReader(str(out)).pages)
    except Exception:  # noqa: BLE001
        pass

    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print("    wrote %s  (%.0f KB%s)"
          % (rel, size_kb, ", %d pages" % pages if pages else ""))

    if pages is not None and pages != n_slides:
        print("    WARNING: %d slides rendered to %d pages -- a slide is "
              "overflowing its page" % (n_slides, pages))
        return 1

    if args.open_after:
        import webbrowser
        webbrowser.open(out.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
