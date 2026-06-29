"""Standalone demo of the Reachy Mini robot baked into the interaction backend.

Start a daemon first: `reachy-mini-daemon` (add `--sim` for simulation), then:
    python -m interaction.robot_demo
The robot greets, then follows the forefront face until Ctrl+C.
"""

import time

from interaction.robot import ReachyMiniRobot


def main() -> None:
    robot = ReachyMiniRobot(greet=True)
    robot.start_following()
    print("Following your face. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        robot.close()


if __name__ == "__main__":
    main()
