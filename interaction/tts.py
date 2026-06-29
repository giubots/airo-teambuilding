#!/usr/bin/env python3
"""Text-to-Speech using OpenAI's gpt-4o-mini-tts with custom personality.

Standalone module — exposes speech(): play a stream of sentences on the
speakers and resolve a Future once all audio has finished playing.
"""

import subprocess
from collections.abc import Generator
from concurrent.futures import Future
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

# Load personality instructions from file
PERSONALITY_FILE = Path(__file__).parent / "personality.txt"
personality = PERSONALITY_FILE.read_text().strip()

client = OpenAI()

# Radio effect: bandpass 300-3000Hz, slight overdrive, compression
RADIO_FILTER = (
    "highpass=f=300,lowpass=f=3000,"
    "acompressor=threshold=-12dB:ratio=4:attack=5:release=50,"
    "asoftclip=type=atan:param=2,"
    "volume=1.5"
)


def _speak(text: str) -> None:
    """Send one sentence to OpenAI TTS and block until playback finishes."""
    if not text.strip():
        return

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions=personality,
        response_format="mp3",
    )

    # Pipe audio through ffmpeg radio filter, then to ffplay
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "quiet",
            "-i", "pipe:0",
            "-af", RADIO_FILTER,
            "-f", "mp3", "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    ffplay = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
        stdin=ffmpeg.stdout,
    )

    ffmpeg.stdin.write(response.content)
    ffmpeg.stdin.close()
    ffplay.wait()


def speech(sentences: Generator[str]) -> Future[str]:
    """Play a stream of sentences sequentially on the speakers.

    Returns immediately with a Future that resolves to the full spoken text
    once every sentence has finished playing. Playback runs in a background
    thread; sentences are spoken in order, one fully before the next starts.
    """
    result: Future[str] = Future()

    def _run() -> None:
        spoken: list[str] = []
        try:
            for sentence in sentences:
                _speak(sentence)
                spoken.append(sentence)
            result.set_result(" ".join(spoken))
        except Exception as exc:  # noqa: BLE001 — surface to the future caller
            result.set_exception(exc)

    Thread(target=_run, daemon=True).start()
    return result


def main() -> None:
    print("OpenAI TTS — type text and press Enter to hear it spoken.")
    print(f"Personality loaded from: {PERSONALITY_FILE}")
    print("Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            text = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Block on this single line until playback completes.
        speech(iter([text])).result()


if __name__ == "__main__":
    main()
