"""Standalone demo: have a little spoken conversation with the Reachy Mini.

Start a daemon first: `reachy-mini-daemon` (add `--sim` for simulation) and set
OPENAI_API_KEY (a .env is auto-loaded), then:
    python -m interaction.conversation
Reachy greets, listens on its mic, transcribes your speech, and replies in its
playful child voice through its own speaker. Ctrl+C to stop.
"""

from interaction.robot import ReachyMiniRobot


def main() -> None:
    robot = ReachyMiniRobot(greet=False)
    try:
        robot.converse(turns=6, listen_secs=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
