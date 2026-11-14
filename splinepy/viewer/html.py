"""Standalone HTML export and notebook display of scenes."""

import html as _html
import pathlib as _pathlib
from contextlib import contextmanager as _contextmanager

from splinepy.viewer import scene as _scene

_JS = _pathlib.Path(__file__).parent / "static" / "viewer.bundle.js"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>html,body{{margin:0;height:100%}}#v{{width:100%;height:100%}}</style>
</head><body><div id="v"></div>
<script type="module">
{js}
const scene = JSON.parse(document.getElementById("scene").textContent);
window.viewer = await splinepyViewer.renderScene(document.getElementById("v"), scene);
</script>
<script type="application/json" id="scene">{scene}</script>
</body></html>
"""


def to_html(scene, title="splinepy viewer"):
    """Returns a self-contained HTML page (three.js is inlined, works offline)."""
    return _PAGE.format(
        title=_html.escape(title),
        js=_JS.read_text(encoding="utf-8"),
        scene=_scene.to_json(scene).replace("</", "<\\/"),
    )


# page of a live session: scenes arrive by long polling, so one tab serves
# every `show()` of the python process. `q` / Esc clear the view and hand
# control back to python, the tab stays for the next scene.
_LIVE = """<!doctype html>
<html><head><meta charset="utf-8"><title>splinepy viewer</title>
<style>html,body{margin:0;height:100%;font:14px sans-serif}#v{width:100%;height:100%}
#msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#666}</style>
</head><body><div id="v"></div><div id="msg"></div>
<script type="module">
{js}
const root = document.getElementById("v"), msg = document.getElementById("msg");
let handle = null, version = 0;
const say = (t) => { msg.textContent = t; msg.style.display = t ? "flex" : "none"; };
const clear = () => { handle?.dispose(); handle = null; window.viewer = undefined; root.replaceChildren(); };
const done = () => { clear(); say("Waiting for the next show() ..."); navigator.sendBeacon("/done"); };
addEventListener("keydown", (e) => { if (handle && (e.key === "q" || e.key === "Escape")) done(); });
addEventListener("pagehide", () => navigator.sendBeacon("/closed"));
let failures = 0;
for (;;) {
  try {
    const m = await (await fetch("/poll?v=" + version)).json();
    failures = 0;
    if (m.scene) {
      version = m.v; clear(); say("");
      handle = window.viewer = await splinepyViewer.renderScene(root, m.scene);
    }
  } catch (e) {
    if (++failures > 2) { clear(); say("Python session ended."); window.close(); break; }
    await new Promise((r) => setTimeout(r, 500));
  }
}
</script></body></html>
"""


class _Session:
    """Local server behind one browser tab, shared by all ``show()`` calls."""

    def __init__(self):
        import http.server
        import threading

        self.cond = threading.Condition()
        self.scene_json, self.version = None, 0
        self.last_poll, self.closed = None, False
        self.done = threading.Event()
        session = self
        page = _LIVE.replace("{js}", _JS.read_text(encoding="utf-8")).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def _reply(self, code, body=b"", ctype="text/plain"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/poll":
                    seen = int(query.split("=")[-1] or 0)
                    self._reply(200, session._poll(seen), "application/json")
                elif path == "/done":
                    session.done.set()
                    self._reply(204)
                elif path == "/closed":
                    session.closed = True
                    session.done.set()
                    self._reply(204)
                else:
                    session.closed = False
                    self._reply(200, page, "text/html; charset=utf-8")

            do_POST = do_GET

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _poll(self, seen, wait=4.0):
        import json
        import time

        with self.cond:
            self.last_poll = time.monotonic()
            self.cond.wait_for(lambda: self.version != seen, timeout=wait)
            self.last_poll = time.monotonic()
            scene = self.scene_json if self.version != seen else None
            return (
                json.dumps(
                    {"v": self.version, "scene": None}
                    if scene is None
                    else {"v": self.version, "scene": "@SCENE@"}
                )
                .replace('"@SCENE@"', scene or "null")
                .encode()
            )

    @property
    def alive(self):
        import time

        return (
            not self.closed
            and self.last_poll is not None
            and time.monotonic() - self.last_poll < 8
        )

    def show(self, scene_json, block, new_window, first_timeout=30):
        import time
        import webbrowser

        reuse = self.alive and not new_window
        self.done.clear()
        with self.cond:
            self.scene_json, self.version = scene_json, self.version + 1
            self.cond.notify_all()
        if not reuse:
            self.closed = False
            webbrowser.open(self.url)
        if not block:
            return
        started = time.monotonic()
        try:
            while not self.done.is_set():
                if (
                    (self.last_poll is None or not self.alive)
                    and time.monotonic() - started > first_timeout
                ):
                    break  # no browser answered
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass


_session = None


def _get_session():
    global _session  # noqa: PLW0603
    if _session is None:
        _session = _Session()
    return _session


class Viewer:
    """Result of :func:`splinepy.viewer.show`. Renders itself in notebooks."""

    def __init__(self, scene, width="100%", height=480, title="splinepy viewer"):
        self.scene = scene
        self.width = width
        self.height = height
        self.title = title

    def to_html(self):
        return to_html(self.scene, title=self.title)

    def save(self, path):
        """Writes a standalone HTML file."""
        _pathlib.Path(path).write_text(self.to_html(), encoding="utf-8")
        return path

    def open(self, block=False, new_window=False):
        """Shows the viewer in the web browser.

        The first call opens a tab; later calls of the python process reuse
        it, which is fast and leaves no tabs behind. ``new_window=True`` opens
        a fresh tab. With ``block=True`` this returns when ``q`` / Esc is
        pressed in the page (it is then cleared, the tab stays), the tab is
        closed or on Ctrl-C, like the vedo window.
        """
        _get_session().show(
            _scene.to_json(self.scene), block=block, new_window=new_window
        )

    def _page(self, width, height, scale=1):
        """Context manager yielding a headless playwright page of the scene."""
        return _headless_page(self.to_html(), width, height, scale)

    def screenshot(self, path, width=800, height=600, camera=None, ui=False, scale=1):
        """Renders a PNG headlessly (needs ``pip install playwright``).

        Overlays like the colorbar and control point labels are included;
        the toggle panel and PNG button are hidden unless ``ui=True``.
        """
        with self._page(width, height, scale) as page:
            if camera is not None:
                page.evaluate("c => viewer.setCamera(c)", camera)
            page.evaluate("on => viewer.setUI(on)", ui)
            page.screenshot(path=str(path))
        return path

    def render_frames(
        self, directory, frames, width=800, height=600, scale=1, ui=False
    ):
        """Renders one PNG per frame into ``directory`` (``00000.png`` ...).

        ``frames`` is a list of camera dicts (see
        :func:`splinepy.viewer.turntable`) and/or scenes, for example the
        deformed state of each step. Returns the written paths.
        """
        directory = _pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        with self._page(width, height, scale) as page:
            page.evaluate("on => viewer.setUI(on)", ui)
            for i, frame in enumerate(frames):
                if "layers" in frame or "panels" in frame:
                    page.evaluate(
                        "s => viewer.setScene(JSON.parse(s))",
                        _scene.to_json(frame),
                    )
                else:
                    page.evaluate("c => viewer.setCamera(c)", frame)
                page.evaluate("viewer.ready()")
                p = directory / f"{i:05d}.png"
                page.screenshot(path=str(p))
                paths.append(p)
        return paths

    def video(self, path, frames, fps=30, width=800, height=600, crf=18):
        """Renders frames and encodes them with ffmpeg (must be on PATH)."""
        import shutil
        import subprocess
        import tempfile

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            self.render_frames(tmp, frames, width, height)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-crf", str(crf), str(path),
                ],
                check=True,
            )  # fmt: skip
        return path

    def _repr_html_(self):
        h = self.height if isinstance(self.height, str) else f"{self.height}px"
        return (
            f'<iframe style="width:{self.width};height:{h};border:0" '
            f'srcdoc="{_html.escape(self.to_html(), quote=True)}"></iframe>'
        )


@_contextmanager
def _headless_page(html, width, height, scale):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "Headless rendering needs playwright: pip install playwright"
            " && playwright install chromium"
        ) from e
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        try:
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=scale,
            )
            page.set_content(html)
            page.wait_for_function("window.viewer !== undefined")
            page.evaluate("viewer.ready()")
            yield page
        finally:
            browser.close()
