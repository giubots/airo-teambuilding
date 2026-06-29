"""Live visualiser for the JPEG-over-WebSocket camera stream.

The camera at ws://192.168.2.203:8123 pushes raw JPEG frames (one binary
message per frame) immediately on connect -- no handshake, no envelope.

This script runs the network in a background thread (matching the project's
synchronous websockets style in interaction/communication.py) and renders
frames with Tkinter on the main thread, which is where all GUI work must
happen. Only the newest frame is kept; if rendering lags behind the stream,
stale frames are dropped so the view stays live.

Usage:
    python3 visualize_stream.py                 # default ws://192.168.2.203:8123
    python3 visualize_stream.py --uri ws://host:port
    python3 visualize_stream.py --save-dir ./captures

Keys:  q / Esc = quit    s = save current frame as PNG
"""

import argparse
import io
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageTk
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException

DEFAULT_URI = "ws://192.168.2.203:8123"


class FrameBuffer:
    """Holds only the most recent decoded frame (newest-wins, drop stale)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image: Image.Image | None = None
        self._seq = 0  # bumps on every new frame so the UI can detect change

    def put(self, image: Image.Image) -> None:
        with self._lock:
            self._image = image
            self._seq += 1

    def get(self, last_seq: int) -> tuple[Image.Image | None, int]:
        """Return (image, seq) only if newer than last_seq, else (None, last_seq)."""
        with self._lock:
            if self._seq == last_seq:
                return None, last_seq
            return self._image, self._seq


class StreamClient(threading.Thread):
    """Background thread: connects, decodes JPEG frames, feeds the FrameBuffer."""

    def __init__(self, uri: str, buffer: FrameBuffer) -> None:
        super().__init__(daemon=True)
        self.uri = uri
        self.buffer = buffer
        self._stop = threading.Event()
        self.frames_recv = 0
        self.status = "starting"

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.status = "connecting"
                with connect(self.uri, max_size=None, open_timeout=10) as ws:
                    self.status = "connected"
                    backoff = 1.0
                    for message in ws:
                        if self._stop.is_set():
                            break
                        if not isinstance(message, (bytes, bytearray)):
                            continue  # ignore any non-binary control text
                        try:
                            img = Image.open(io.BytesIO(message))
                            img.load()  # force decode now, inside the net thread
                        except Exception:
                            continue  # skip a corrupt/partial frame
                        self.buffer.put(img)
                        self.frames_recv += 1
            except (WebSocketException, OSError) as exc:
                if self._stop.is_set():
                    break
                self.status = f"reconnecting ({type(exc).__name__})"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 10.0)
        self.status = "stopped"


class Viewer:
    """Tkinter window that renders the newest frame and an FPS/status overlay."""

    def __init__(self, root: tk.Tk, client: StreamClient, buffer: FrameBuffer,
                 save_dir: Path) -> None:
        self.root = root
        self.client = client
        self.buffer = buffer
        self.save_dir = save_dir
        self.last_seq = 0
        self.current_image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None

        # FPS measured over a sliding window of displayed frames.
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self._fps = 0.0

        root.title("Camera stream")
        root.geometry("800x600")
        root.configure(bg="black")
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        root.bind("<q>", lambda _e: self._quit())
        root.bind("<Escape>", lambda _e: self._quit())
        root.bind("<s>", lambda _e: self._save())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        self._tick()

    def _tick(self) -> None:
        image, seq = self.buffer.get(self.last_seq)
        if image is not None:
            self.last_seq = seq
            self.current_image = image
            self._fps_count += 1
            now = time.monotonic()
            elapsed = now - self._fps_t0
            if elapsed >= 0.5:
                self._fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_t0 = now
        self._render()
        # ~33 ms cadence (~30 Hz); the stream's own rate caps actual display fps.
        self.root.after(30, self._tick)

    def _render(self) -> None:
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        if self.current_image is None:
            self.canvas.create_text(
                cw // 2, ch // 2, fill="gray",
                text=f"Waiting for frames...\n{self.client.status}",
                font=("TkDefaultFont", 14), justify="center",
            )
            return

        # Scale to fit the window while preserving aspect ratio.
        iw, ih = self.current_image.size
        scale = min(cw / iw, ch / ih)
        new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
        resample = Image.NEAREST if scale > 1 else Image.BILINEAR
        shown = self.current_image.resize(new_size, resample)
        self._tk_image = ImageTk.PhotoImage(shown)
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_image)

        overlay = (
            f"{iw}x{ih}  |  {self._fps:4.1f} fps  |  "
            f"{self.client.frames_recv} frames  |  {self.client.status}"
        )
        # Drop shadow then text, so the overlay stays readable on any frame.
        self.canvas.create_text(11, 11, anchor="nw", fill="black", text=overlay,
                                font=("TkDefaultFont", 11))
        self.canvas.create_text(10, 10, anchor="nw", fill="#00ff66", text=overlay,
                                font=("TkDefaultFont", 11))

    def _save(self) -> None:
        if self.current_image is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f.png")
        path = self.save_dir / name
        self.current_image.save(path)
        print(f"Saved {path}")

    def _quit(self) -> None:
        self.client.stop()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI,
                        help=f"WebSocket URI (default: {DEFAULT_URI})")
    parser.add_argument("--save-dir", default="captures", type=Path,
                        help="Directory for 's' key snapshots (default: ./captures)")
    args = parser.parse_args()

    buffer = FrameBuffer()
    client = StreamClient(args.uri, buffer)
    client.start()

    root = tk.Tk()
    Viewer(root, client, buffer, args.save_dir)
    try:
        root.mainloop()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
