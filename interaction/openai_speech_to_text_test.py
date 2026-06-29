import asyncio
import json
import os
import tempfile
import wave
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI


SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
ASR_MODEL = "gpt-4o-mini-transcribe"
PARSER_MODEL = "gpt-4o-mini"
SUPPORTED_INTENTS = {"find_object", "pick_object", "place_object", "stop", "confirm", "reject"}


def load_environment() -> None:
    base_dir = Path(__file__).parent
    load_dotenv(base_dir / ".env")
    load_dotenv(base_dir / ".env.example")


def get_api_key() -> str:
    api_key = input("Please enter your OpenAI API key (press Enter to use .env): ").strip()
    if api_key:
        return api_key
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    raise ValueError("OPENAI_API_KEY is missing. Enter key in terminal or set it in .env/.env.example.")


def record_wav() -> Path:
    print(f"[System] Recording {RECORD_SECONDS} seconds...")
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
    return wav_path


def transcribe(client: OpenAI, wav_path: Path) -> str:
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=ASR_MODEL,
            file=f,
            language="en",
        )
    return result.text.strip()


def parse_intent(client: OpenAI, transcript: str) -> dict:
    response = client.chat.completions.create(
        model=PARSER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent parser for a robotic assistant.\n"
                    "Supported intents: find_object, pick_object, place_object, stop, confirm, reject.\n"
                    "Return JSON only with keys: intent, object.\n"
                    "Use object=null when not applicable."
                ),
            },
            {"role": "user", "content": f'Input:\n"{transcript}"\n\nReturn the JSON object only.'},
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    intent = data.get("intent")
    if intent not in SUPPORTED_INTENTS:
        raise ValueError(f"Unsupported intent: {intent}")
    if "object" not in data:
        data["object"] = None
    return data


def is_empty_or_null(text: str) -> bool:
    return text.strip().lower() in {"", "null", "none", "n/a", "[silence]"}


async def run_once(client: OpenAI) -> None:
    input("[System] Press Enter to record 5 seconds...")
    wav_path = await asyncio.to_thread(record_wav)
    try:
        transcript = await asyncio.to_thread(transcribe, client, wav_path)
        print(f"[Transcript] {transcript}")

        if is_empty_or_null(transcript):
            print("Sorry, I did not catch you. Please repeat.")
            return

        try:
            command = await asyncio.to_thread(parse_intent, client, transcript)
        except ValueError:
            print("Sorry, I did not catch what you need. Please repeat.")
            return

        print(json.dumps(command, ensure_ascii=False, indent=2))
    finally:
        wav_path.unlink(missing_ok=True)


async def main() -> None:
    load_environment()
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    while True:
        await run_once(client)
        if input("Run another STT cycle? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    asyncio.run(main())
