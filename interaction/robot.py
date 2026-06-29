"""Reachy Mini robot control, baked into the interaction package.

This wraps a real Reachy Mini (or simulation) so the interaction backend can
give the robot expressive behaviour: a greeting, head gestures, an antenna
"happy" wiggle, and continuous face following with the forefront person.

A daemon must be running (`reachy-mini-daemon`, add `--sim` for simulation).
The connection auto-detects Lite vs Wireless and localhost vs network.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

try:  # optional offline text-to-speech
    import pyttsx3

    _tts = pyttsx3.init()
except Exception:  # pragma: no cover - missing pyttsx3 or audio device
    _tts = None

Face = Tuple[int, int, int, int]


@dataclass
class DoneLook:
    success: bool
    img: Optional["np.ndarray"]


class FaceTracker:
    """Pick the forefront face with lock-on hysteresis and smooth the aim.

    aim_down_frac shifts the aim below the box center (0 = center, 0.5 = bottom)
    so the head centers on the face instead of the brow.
    """

    def __init__(self, smooth: float = 0.4, lock_gate: int = 350,
                 aim_down_frac: float = 0.45, detect_width: int = 640) -> None:
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.smooth = smooth
        self.lock_gate = lock_gate
        self.aim_down_frac = aim_down_frac
        self.detect_width = detect_width
        self.sx: Optional[float] = None
        self.sy: Optional[float] = None

    def detect(self, frame) -> List[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        scale = self.detect_width / gray.shape[1]
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        boxes = self.detector.detectMultiScale(small, 1.15, 6, minSize=(30, 30))
        return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                for (x, y, w, h) in boxes]

    def update(self, faces: List[Face]) -> Optional[Tuple[float, float]]:
        if not faces:
            return None
        cand = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        pick = cand[0]
        if self.sx is not None:
            near = [f for f in cand
                    if abs(f[0] + f[2] / 2 - self.sx) < self.lock_gate
                    and abs(f[1] + f[3] / 2 - self.sy) < self.lock_gate]
            if near:
                pick = near[0]
        x, y, w, h = pick
        u = x + w / 2
        v = y + h * (0.5 + self.aim_down_frac)
        a = self.smooth
        self.sx = u if self.sx is None else a * u + (1 - a) * self.sx
        self.sy = v if self.sy is None else a * v + (1 - a) * self.sy
        return self.sx, self.sy


class ReachyMiniRobot:
    """Expressive control of a Reachy Mini for the interaction backend."""

    logger: logging.Logger

    def __init__(self, greet: bool = True) -> None:
        self.logger = logging.getLogger(__name__)
        self.mini = ReachyMini(media_backend="default")
        self.mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
        self._tracker = FaceTracker()
        self._follow = False
        self._th: Optional[threading.Thread] = None
        if greet:
            self.greet()

    def say(self, text: str) -> None:
        self.logger.info("Reachy says: %s", text)
        if _tts is not None:
            _tts.say(text)
            _tts.runAndWait()

    def happy(self) -> None:
        self.mini.goto_target(antennas=[0.5, -0.5], duration=0.4)
        self.mini.goto_target(antennas=[-0.5, 0.5], duration=0.4)
        self.mini.goto_target(antennas=[0.0, 0.0], duration=0.4)

    def greet(self) -> None:
        self.say("Hello!")
        self.happy()

    def look(self) -> DoneLook:
        """Grab a camera frame (RGB). Used by the interaction 'look' step."""
        try:
            return DoneLook(True, self.mini.media.get_frame())
        except Exception as e:  # camera not ready
            self.logger.warning("look failed: %s", e)
            return DoneLook(False, None)

    def start_following(self) -> None:
        if self._follow:
            return
        self._follow = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop_following(self) -> None:
        self._follow = False
        if self._th:
            self._th.join(timeout=1.0)
        self.mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

    def _loop(self) -> None:
        miss = 0
        while self._follow:
            frame = self.mini.media.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            aim = self._tracker.update(self._tracker.detect(frame))
            miss = 0 if aim else miss + 1
            if self._tracker.sx is not None and miss < 8:
                pose = self.mini.look_at_image(int(self._tracker.sx),
                                               int(self._tracker.sy),
                                               duration=0, perform_movement=False)
                self.mini.set_target(head=pose)
            time.sleep(0.03)

    def close(self) -> None:
        self.stop_following()
        self.mini.__exit__(None, None, None)
