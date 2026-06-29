import asyncio
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from openai import OpenAI


SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
ASR_MODEL = "gpt-4o-mini-transcribe"

# RMS threshold below which audio is considered silence (0–32767 scale)
SILENCE_THRESHOLD = 200

EMPTY_TRANSCRIPTS = {"", "null", "none", "n/a", "[silence]", "[blank_audio]"}


def get_api_key() -> str:
    key = input("Please enter your OpenAI API key: ").strip()
    if not key:
        raise ValueError("API key cannot be empty.")
    return key


def record_wav() -> Path:
    print(f"[System] Recording {RECORD_SECONDS} seconds... (speak now)")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    wav_path = Path(tmp.name)

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    return wav_path, audio


def is_silent(audio: np.ndarray) -> bool:
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms < SILENCE_THRESHOLD


def transcribe(client: OpenAI, wav_path: Path) -> str:
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=f,
            language="en",
        )
    return result.text.strip()


def is_empty_transcript(text: str) -> bool:
    return text.lower() in EMPTY_TRANSCRIPTS


async def run_once(client: OpenAI) -> None:
    input("[System] Press Enter to start recording...")
    wav_path, audio = await asyncio.to_thread(record_wav)
    try:
        if is_silent(audio):
            print("Pardon me")
            return

        transcript = await asyncio.to_thread(transcribe, client, wav_path)

        if is_empty_transcript(transcript):
            print("Pardon me")
            return

        print(f"[Transcription] {transcript}")
    finally:
        wav_path.unlink(missing_ok=True)


async def main() -> None:
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    while True:
        await run_once(client)
        if input("Run another STT round? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    asyncio.run(main())
