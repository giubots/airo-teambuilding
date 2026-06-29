#!/usr/bin/env python3
"""Manual test for interaction.tts.speech(). Run from repo root: python test_tts.py"""

import time

from interaction.tts import speech


def sentences():
    for s in ["Wow, hello there!", "This is the second sentence.", "And the last one!"]:
        yield s


if __name__ == "__main__":
    print("Starting playback in background; future should resolve when audio ends.")
    t0 = time.time()
    fut = speech(sentences())
    print(f"speech() returned immediately after {time.time() - t0:.3f}s")
    result = fut.result()  # blocks until all sentences played
    print(f"Done after {time.time() - t0:.1f}s. Full text: {result!r}")

    # Interrupt demo: start a long stream, stop after 4 seconds.
    print("\nInterrupt demo: stopping after 4s...")
    long = (f"Sentence number {i}." for i in range(20))
    fut = speech(long)
    time.sleep(4)
    fut.stop()
    print(f"Stopped. Spoken before interrupt: {fut.result()!r}")
