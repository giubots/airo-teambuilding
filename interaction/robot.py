"""Reachy Mini robot control, baked into the interaction package.

This wraps a real Reachy Mini (or simulation) so the interaction backend can
give the robot expressive behaviour: a greeting, head gestures, an antenna
"happy" wiggle, and continuous face following with the forefront person.

A daemon must be running (`reachy-mini-daemon`, add `--sim` for simulation).
The connection auto-detects Lite vs Wireless and localhost vs network.
"""

import logging
import math
import os
import queue
import random
import base64
import tempfile
import threading
import time
import wave
from concurrent.futures import Future
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from dotenv import load_dotenv

# Load OPENAI_API_KEY so the cloud "coral" voice is used (else falls back to
# pyttsx3). Look at cwd and the repo root so it works regardless of launch dir.
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_PERSONALITY_FILE = Path(__file__).with_name("personality.txt")
_PERSONALITY = _PERSONALITY_FILE.read_text(encoding="utf-8").strip() if _PERSONALITY_FILE.exists() else ""

# Loudness: peak-normalise every clip near full-scale + extra gain so the small
# robot speaker is audible (REACHY_TTS_GAIN tunes it; 1.0 = normalise only).
_TARGET_PEAK = 0.97
_GAIN = float(os.getenv("REACHY_TTS_GAIN", "1.6"))


def _sentence_end(buf: str) -> int:
    """Index just past the first sentence-ending . ! ? in buf, else -1."""
    for i, c in enumerate(buf):
        if c in ".!?" and (i + 1 >= len(buf) or buf[i + 1] in " \n\t"):
            return i + 1
    return -1


_BODY_PEAK_DEG = 11.0   # gentle body swivel each way in 'thinking' mode (limits head motion)
_DANCE_PERIOD = 5.0     # seconds per full left-right-left sway (relaxed)


def _swivel_yaw(t: float) -> float:
    """Body yaw (rad) for a smooth left-right sway at time t."""
    return math.radians(_BODY_PEAK_DEG) * math.sin(2 * math.pi * t / _DANCE_PERIOD)


def _yaw_pose(pose, yaw_rad: float):
    """Rotate a 4x4 head pose about the vertical axis.

    Not used to keep gaze on the participant — with automatic_body_yaw the IK
    already holds the world-frame head pose while the body swivels.
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return pose @ rz


def _make_elevator_wav(path: str, secs: float = 8.0, rate: int = 16000) -> str:
    """Write a soft, loopable elevator-music clip (simple chord arpeggio)."""
    chords = [(261, 329, 392), (220, 277, 329), (174, 220, 261), (196, 247, 392)]
    beat = secs / (len(chords) * 2)
    out = []
    for ch in chords * 2:
        for f in ch:
            t = np.linspace(0, beat, int(rate * beat), endpoint=False)
            env = np.minimum(1.0, 8 * np.minimum(t, beat - t) / beat)
            out.append(0.18 * env * np.sin(2 * math.pi * f * t))
    pcm = (np.clip(np.concatenate(out), -1, 1) * 32767).astype(np.int16)
    with closing(wave.open(path, "wb")) as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path

_openai = None
if os.getenv("OPENAI_API_KEY"):
    try:  # cloud voice (openai-tts branch); falls back to pyttsx3 below
        from openai import OpenAI

        _openai = OpenAI()
    except Exception:
        _openai = None

try:  # optional offline text-to-speech
    import pyttsx3

    _tts = pyttsx3.init()
except Exception:  # pragma: no cover - missing pyttsx3 or audio device
    _tts = None

Face = Tuple[int, int, int, int]


def _roll_pose(pose, roll_rad: float):
    """Add a head roll (rotation about the look axis) to a 4x4 head pose."""
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    return pose @ rx


class IdleTilt:
    """Occasional relaxed head tilt: once in a while, ease over and back."""

    def __init__(self, peak_deg: float = 13.0, dur: float = 3.0,
                 gap: Tuple[float, float] = (10.0, 18.0)) -> None:
        self.peak = math.radians(peak_deg)
        self.dur = dur
        self.gap = gap
        self._t0: Optional[float] = None
        self._dir = 1.0
        self._next = time.time() + random.uniform(*gap)

    def roll(self, still: bool) -> float:
        """Roll (rad) to apply now: a single eased tilt, else 0 while waiting."""
        now = time.time()
        if self._t0 is not None:
            t = now - self._t0
            if t >= self.dur:
                self._t0, self._next = None, now + random.uniform(*self.gap)
                return 0.0
            return self.peak * self._dir * math.sin(math.pi * t / self.dur)
        if still and now >= self._next:
            self._t0, self._dir = now, random.choice([-1.0, 1.0])
        return 0.0


class EarWiggle:
    """Occasional ear wiggle: a brief burst, then a quiet cooldown."""

    def __init__(self, dur: float = 0.9, gap: Tuple[float, float] = (5.0, 10.0)) -> None:
        self.dur = dur
        self.gap = gap
        self._t0: Optional[float] = None
        self._next = 0.0

    def antennas(self, talking: bool) -> List[float]:
        now = time.time()
        if self._t0 is not None:
            t = now - self._t0
            if t >= self.dur:
                self._t0, self._next = None, now + random.uniform(*self.gap)
                return [0.0, 0.0]
            w = math.sin(2 * math.pi * 3.0 * t) * 0.4 * math.sin(math.pi * t / self.dur)
            return [w, -w]
        if talking and now >= self._next:
            self._t0 = now
        return [0.0, 0.0]


@dataclass
class DoneLook:
    success: bool
    img: Optional["np.ndarray"]


class FaceTracker:
    """Pick the forefront face with lock-on hysteresis and smooth the aim.

    aim_down_frac shifts the aim below the box center (0 = center, 0.5 = bottom)
    so the head centers on the face instead of the brow.
    """

    def __init__(self, smooth: float = 0.2, lock_gate: int = 320,
                 aim_down_frac: float = 0.45, detect_width: int = 1920,
                 hold_secs: float = 2.0, deadzone: int = 14) -> None:
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        try:  # MediaPipe: robust to tilt/profile, ~9ms; full range (model 1)
            from mediapipe.python.solutions import face_detection as mpfd
            self._mp = mpfd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            self._mp = None
        self.smooth = smooth
        self.lock_gate = lock_gate
        self.aim_down_frac = aim_down_frac
        self.detect_width = detect_width
        self.hold_secs = hold_secs
        self.deadzone = deadzone
        self.sx: Optional[float] = None
        self.sy: Optional[float] = None
        self._still_since: Optional[float] = None
        self._seen = 0.0

    def is_still(self, settle: float = 1.5) -> bool:
        """True once the locked face has barely moved for `settle` seconds."""
        return self._still_since is not None and (time.time() - self._still_since) > settle

    def has_lock(self) -> bool:
        """Locked or coasting: aim valid and last seen within hold window."""
        return self.sx is not None and (time.time() - self._seen) < self.hold_secs

    def detect(self, frame) -> List[Face]:
        """Faces in full-frame pixels via MediaPipe (Haar+equalize fallback)."""
        h, w = frame.shape[:2]
        if self._mp is not None:
            res = self._mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            faces: List[Face] = []
            for d in (res.detections or []):
                b = d.location_data.relative_bounding_box
                faces.append((int(b.xmin * w), int(b.ymin * h),
                              int(b.width * w), int(b.height * h)))
            return faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # SDK frames are BGR
        scale = min(1.0, self.detect_width / gray.shape[1])
        small = cv2.equalizeHist(cv2.resize(gray, None, fx=scale, fy=scale))
        boxes = self.detector.detectMultiScale(small, 1.1, 6, minSize=(80, 80))
        return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                for (x, y, w, h) in boxes]

    def update(self, faces: List[Face]) -> Optional[Tuple[float, float]]:
        """Sticky-lock the same face; coast on the last aim during brief losses."""
        now = time.time()
        if not faces:
            if self.sx is not None and now - self._seen > self.hold_secs:
                self.sx = self.sy = self._still_since = None
            return (self.sx, self.sy) if self.sx is not None else None
        if self.has_lock():
            near = [f for f in faces
                    if abs(f[0] + f[2] / 2 - self.sx) < self.lock_gate
                    and abs(f[1] + f[3] / 2 - self.sy) < self.lock_gate]
            pick = min(near, key=lambda f: (f[0] + f[2] / 2 - self.sx) ** 2
                       + (f[1] + f[3] / 2 - self.sy) ** 2) if near \
                else max(faces, key=lambda f: f[2] * f[3])
        else:
            pick = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = pick
        u = x + w / 2
        v = y + h * (0.5 + self.aim_down_frac)
        if self.sx is None:
            self.sx, self.sy, self._still_since = u, v, now
        else:
            a = self.smooth
            self.sx += a * (u - self.sx)
            self.sy += a * (v - self.sy)
            if abs(u - self.sx) > 25 or abs(v - self.sy) > 25:
                self._still_since = now
        self._seen = now  # moved: reset stillness timer
        return self.sx, self.sy


class ReachyMiniRobot:
    """Expressive control of a Reachy Mini for the interaction backend."""

    logger: logging.Logger

    def __init__(self, greet: bool = True) -> None:
        self.logger = logging.getLogger(__name__)
        self.mini = ReachyMini(media_backend="default")
        self.mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)
        self._tracker = FaceTracker()
        self._tilt = IdleTilt()
        self._ears = EarWiggle()
        self._follow = False
        self._th: Optional[threading.Thread] = None
        self._speaking = threading.Lock()
        self._thinking = False
        self._dance_t0 = 0.0
        self._music_stop: Optional[threading.Event] = None
        if greet:
            self.greet()

    def say(self, text: str, block: bool = True) -> None:
        """Speak through the robot's own speaker (OpenAI TTS, else offline).

        block=False offloads synthesis + playback to a background thread so the
        face-following loop never freezes; overlapping calls are dropped.
        """
        self.logger.info("Reachy says: %s", text)
        if not text.strip():
            return
        if not block:
            if self._speaking.acquire(blocking=False):
                def run():
                    try:
                        self._do_say(text)
                    finally:
                        self._speaking.release()
                threading.Thread(target=run, daemon=True).start()
            return
        self._do_say(text)

    def _do_say(self, text: str) -> None:
        path = os.path.join(tempfile.gettempdir(), "reachy_say.wav")
        if not self._synth_wav(text, path):
            return
        try:
            self.mini.media.play_sound(path)
            time.sleep(self._wav_seconds(path) + 0.2)
        except Exception as e:  # pragma: no cover - audio backend missing
            self.logger.warning("play_sound failed: %s", e)

    @staticmethod
    def _wav_seconds(path: str) -> float:
        try:
            with closing(wave.open(path, "rb")) as w:
                ch, sw, rate, n = (w.getnchannels(), w.getsampwidth(),
                                   w.getframerate(), w.getnframes())
            if 0 < n < rate * 600:
                secs = n / float(rate or 1)
            else:  # OpenAI WAVs use an unknown-length sentinel; use file size
                secs = max(0, os.path.getsize(path) - 44) / float(rate * ch * sw or 1)
            return min(secs, 60.0)
        except Exception:
            return 1.5

    @staticmethod
    def _boost_wav(path: str) -> None:
        """Peak-normalise a 16-bit WAV in place so the robot speaker is loud enough."""
        try:
            with closing(wave.open(path, "rb")) as w:
                ch, sw, rate, n = (w.getnchannels(), w.getsampwidth(),
                                   w.getframerate(), w.getnframes())
                raw = w.readframes(n if 0 < n < rate * 600 else rate * 60)
            if sw != 2 or not raw:
                return
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            peak = float(np.max(np.abs(a))) or 1.0
            a = np.clip(a * (_TARGET_PEAK * 32767.0 / peak) * _GAIN, -32768, 32767)
            with closing(wave.open(path, "wb")) as w:
                w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
                w.writeframes(a.astype(np.int16).tobytes())
        except Exception:
            pass

    def _synth_wav(self, text: str, path: str) -> bool:
        if _openai is not None:
            try:
                r = _openai.audio.speech.create(model="gpt-4o-mini-tts", voice="coral",
                                                input=text, instructions=_PERSONALITY,
                                                response_format="wav")
                r.write_to_file(path)
                self._boost_wav(path)
                return True
            except Exception:
                pass
        if _tts is not None:
            _tts.save_to_file(text, path)
            _tts.runAndWait()
            self._boost_wav(path)
            return os.path.exists(path)
        return False

    def record(self, seconds: float = 3.0) -> np.ndarray:
        """Capture mono float32 audio from the mic array (scaffolding for STT)."""
        self.mini.media.start_recording()
        chunks, end = [], time.time() + seconds
        while time.time() < end:
            s = self.mini.media.get_audio_sample()
            if s is not None:
                a = np.asarray(s, dtype=np.float32)
                chunks.append(a.mean(axis=1) if a.ndim > 1 else a)
            else:
                time.sleep(0.01)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def listen(self, seconds: float = 5.0) -> str:
        """Record from the robot mic and transcribe it (gpt-4o-mini-transcribe)."""
        samples = self.record(seconds)
        if samples.size == 0 or _openai is None:
            return ""
        try:
            sr = max(1, self.mini.media.get_input_audio_samplerate())
        except Exception:
            sr = 16000
        path = os.path.join(tempfile.gettempdir(), "reachy_listen.wav")
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        with closing(wave.open(path, "wb")) as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        try:
            with open(path, "rb") as f:
                r = _openai.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", file=f, language="en")
            return (r.text or "").strip()
        except Exception:
            return ""

    def reply(self, text: str) -> str:
        """One-or-two-sentence reply in the child personality (rolling history)."""
        self._chat_init()
        self._chat.append({"role": "user", "content": text})
        if _openai is None:
            out = "Wow, that's so cool! Tell me more!"
        else:
            try:
                r = _openai.chat.completions.create(model="gpt-4o-mini", temperature=0.8,
                                                    max_tokens=80, messages=self._chat)
                out = (r.choices[0].message.content or "").strip()
            except Exception:
                out = "Hmm, I didn't quite get that. Say it again?"
        self._remember(out)
        return out

    def reply_stream(self, text: str):
        """Stream the reply, yielding each sentence as soon as it completes."""
        self._chat_init()
        self._chat.append({"role": "user", "content": text})
        if _openai is None:
            out = "Wow, that's so cool! Tell me more!"
            self._remember(out)
            yield out
            return
        buf, full = "", []
        try:
            stream = _openai.chat.completions.create(model="gpt-4o-mini", temperature=0.8,
                                                     max_tokens=80, messages=self._chat,
                                                     stream=True)
            for chunk in stream:
                buf += chunk.choices[0].delta.content or ""
                while (cut := _sentence_end(buf)) >= 0:
                    s, buf = buf[:cut].strip(), buf[cut:].lstrip()
                    if s:
                        full.append(s)
                        yield s
            if buf.strip():
                full.append(buf.strip())
                yield buf.strip()
        except Exception:
            yield "Hmm, I didn't quite get that. Say it again?"
        self._remember(" ".join(full))

    def _chat_init(self) -> None:
        if not hasattr(self, "_chat"):
            sys_p = (_PERSONALITY + " Keep replies to one or two short sentences. "
                     "Use plain words only, no emoji. "
                     "You are a small robot named Reachy talking face to face.")
            self._chat = [{"role": "system", "content": sys_p}]

    def _remember(self, out: str) -> None:
        self._chat.append({"role": "assistant", "content": out})
        if len(self._chat) > 17:
            self._chat = [self._chat[0]] + self._chat[-16:]

    def speak_stream(self, sentences) -> Future:
        """Speak a sentence stream gapless: synth ahead while playing in order."""
        fut: Future = Future()
        pipe: queue.Queue = queue.Queue(maxsize=2)

        def produce():
            i = 0
            for s in sentences:
                if not s.strip():
                    continue
                p = os.path.join(tempfile.gettempdir(), f"reachy_stream_{i % 4}.wav")
                i += 1
                if self._synth_wav(s, p):
                    pipe.put((s, p))
            pipe.put(None)

        def consume():
            spoken = []
            self._speaking.acquire()  # pause mic STT while speaking
            try:
                while (item := pipe.get()) is not None:
                    s, p = item
                    try:
                        self.mini.media.play_sound(p)
                        time.sleep(self._wav_seconds(p) + 0.15)
                    except Exception:
                        pass
                    spoken.append(s)
                fut.set_result(" ".join(spoken))
            finally:
                self._speaking.release()

        threading.Thread(target=produce, daemon=True).start()
        threading.Thread(target=consume, daemon=True).start()
        return fut

    def converse(self, turns: int = 6, listen_secs: float = 5.0) -> None:
        """Short spoken back-and-forth: listen, think, speak — `turns` times."""
        self.say("Hi! I'm Reachy. Talk to me!", block=True)
        for _ in range(turns):
            heard = self.listen(listen_secs)
            if not heard:
                self.say("I didn't catch that. Try again!", block=True)
                continue
            self.logger.info("Heard: %s", heard)
            self.speak_stream(self.reply_stream(heard)).result()
        self.say("Bye bye! That was fun!", block=True)


    def happy(self) -> None:
        self.mini.goto_target(antennas=[0.5, -0.5], duration=0.4)
        self.mini.goto_target(antennas=[-0.5, 0.5], duration=0.4)
        self.mini.goto_target(antennas=[0.0, 0.0], duration=0.4)

    def greet(self) -> None:
        self.say("Hello!", block=False)
        self.happy()

    def look(self) -> DoneLook:
        """Grab a camera frame (BGR). Used by the interaction 'look' step."""
        try:
            return DoneLook(True, self.mini.media.get_frame())
        except Exception as e:  # camera not ready
            self.logger.warning("look failed: %s", e)
            return DoneLook(False, None)

    def capture_base64(self, fmt: str = "jpeg", quality: int = 90,
                       data_url: bool = False) -> Optional[str]:
        """One camera frame as a base64 string (JPEG/PNG), or data URL; None if no frame."""
        frame = self.mini.media.get_frame()
        if frame is None:
            return None
        ext = ".png" if fmt.lower() == "png" else ".jpg"
        params = [] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        ok, buf = cv2.imencode(ext, frame, params)  # SDK frames are BGR, as cv2 expects
        if not ok:
            return None
        b64 = base64.b64encode(buf).decode("ascii")
        mime = "png" if ext == ".png" else "jpeg"
        return f"data:image/{mime};base64,{b64}" if data_url else b64

    def start_following(self, chat: bool = False) -> None:
        if self._follow:
            return
        self._follow = True
        try:
            self.mini.media.start_recording()  # mic on for talk detection (DoA)
        except Exception as e:  # pragma: no cover - no audio device
            self.logger.warning("mic start failed: %s", e)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        if chat:
            self._chat_stop = threading.Event()
            threading.Thread(target=self._chat_loop, daemon=True).start()

    def stop_following(self) -> None:
        self._follow = False
        self.set_thinking(False)
        if getattr(self, "_chat_stop", None):
            self._chat_stop.set()
        if self._th:
            self._th.join(timeout=1.0)
        try:
            self.mini.media.stop_recording()
        except Exception:
            pass
        self.mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

    def _chat_loop(self, listen_secs: float = 4.0) -> None:
        """Converse in the background while the head keeps tracking the face.

        Pauses the mic while Reachy speaks, and streams the reply
        sentence-by-sentence so audio starts almost immediately. While
        thinking/dance mode is on the whole conversation stays silent: no
        greeting, listening, replying or speaking.
        """
        greeted = False
        while self._follow and not self._chat_stop.is_set():
            if self._thinking or self._speaking.locked():
                time.sleep(0.15)
                continue
            if not greeted:  # greet only once we're actually allowed to speak
                self.say("Hi! I'm Reachy. Talk to me!", block=False)
                greeted = True
                continue
            heard = self.listen(listen_secs)
            if self._chat_stop.is_set() or self._thinking:
                continue
            if not heard:
                continue
            self.logger.info("Heard: %s", heard)
            self.speak_stream(self.reply_stream(heard)).result()

    def set_thinking(self, on: bool) -> None:
        """Toggle 'thinking/doing' mode: swivel the body + loop elevator music.

        The face-following loop keeps the gaze engaged: the world-frame head
        pose stays locked on the participant (the IK holds it while the body
        swivels), so Reachy looks busy-but-attentive while you wait on it.
        Speech is suppressed while thinking so the modes stay separate.
        """
        on = bool(on)
        if on == self._thinking:
            return
        self._thinking = on
        if on:
            self._dance_t0 = time.time()
            self._music_stop = threading.Event()
            path = os.path.join(tempfile.gettempdir(), "reachy_elevator.wav")
            if not os.path.exists(path):
                _make_elevator_wav(path)
            stop = self._music_stop

            def _music():
                try:
                    while not stop.is_set():
                        try:
                            self.mini.media.play_sound(path)
                        except Exception:
                            pass
                        if stop.wait(8.0):  # wake the instant we leave thinking mode
                            break
                finally:
                    try:
                        self.mini.media.stop_playing()  # silence the last clip
                    except Exception:
                        pass
            threading.Thread(target=_music, daemon=True).start()
        elif self._music_stop is not None:
            self._music_stop.set()
            try:
                self.mini.media.stop_playing()  # cut the music now, don't wait it out
            except Exception:
                pass

    def _loop(self) -> None:
        ax = ay = None  # last applied aim (deadzone gate)
        while self._follow:
            frame = self.mini.media.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            self._tracker.update(self._tracker.detect(frame))

            # Ears wiggle now and then while the focused (front) person talks.
            try:
                doa = self.mini.media.get_DoA()
            except Exception:
                doa = None
            talking = (doa is not None and doa[1] and self._tracker.has_lock()
                       and abs(doa[0] - math.pi / 2) < 0.9)
            antennas = self._ears.antennas(talking)

            if self._tracker.has_lock():
                if ax is None or abs(self._tracker.sx - ax) > self._tracker.deadzone \
                        or abs(self._tracker.sy - ay) > self._tracker.deadzone:
                    ax, ay = self._tracker.sx, self._tracker.sy
                pose = self.mini.look_at_image(int(ax), int(ay),
                                               duration=0, perform_movement=False)
                # Sitting still: occasionally tilt the head, ease over and back.
                roll = self._tilt.roll(self._tracker.is_still())
                if roll:
                    pose = _roll_pose(pose, roll)
                byaw = 0.0
                if self._thinking:  # swivel the body; world-frame head stays on the face
                    byaw = _swivel_yaw(time.time() - self._dance_t0)
                self.mini.set_target(head=pose, antennas=antennas, body_yaw=byaw)
            time.sleep(0.03)

    def close(self) -> None:
        self.stop_following()
        self.mini.__exit__(None, None, None)
