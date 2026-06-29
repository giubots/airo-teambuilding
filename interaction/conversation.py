"""Standalone demo: Reachy follows your face AND chats at the same time.

Start a daemon first: `reachy-mini-daemon` (add `--sim` for simulation) and set
OPENAI_API_KEY (a .env is auto-loaded), then:
    python -m interaction.conversation
Reachy tracks the forefront face (idle tilt + ear wiggles) while listening on
its mic, transcribing, and replying in its child voice through its speaker.
Ctrl+C to stop.
"""

import time

from interaction.robot import ReachyMiniRobot


def main() -> None:
    robot = ReachyMiniRobot(greet=False)
    robot.start_following(chat=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        robot.close()


if __name__ == "__main__":
    main()
