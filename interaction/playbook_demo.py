"""Demo: Lionel runs the scripted 'backpack' playbook while following your face.

Start a daemon first: `reachy-mini-daemon` (add `--sim` for simulation), then:
    python -m interaction.playbook_demo

Lionel tracks the forefront face the whole time. The interaction is strictly
turn-based - it is either listening/talking OR thinking, never both:

  1. Say 'Hi Lionel!' to wake it.
  2. It asks what to grab from your backpack and listens.
  3. It extracts the object as {noun, adjective} JSON (saved to
     backpack_request.json and securely copied to the configured server over
     SSH) and says 'Okay, let me search for ...'.
  4. It enters thinking/dance mode to 'pick' the item.
  5. Press Enter here to end thinking; it says 'Happy to help!' and waits for
     the next 'Hi Lionel!'. Ctrl+C quits.

Needs media (hardware: `reachy-mini-daemon`) and OPENAI_API_KEY (.env loaded).
"""

from interaction.robot import ReachyMiniRobot


def main() -> None:
    robot = ReachyMiniRobot(greet=False)
    robot.start_following(playbook=True)
    print("Playbook running. Say 'Hi Lionel!'. "
          "Press Enter to end thinking mode; Ctrl+C to quit.")
    try:
        while True:
            input()  # each Enter ends thinking mode (exit-only during playbook)
            robot.toggle_thinking()
    except (KeyboardInterrupt, EOFError):
        robot.close()


if __name__ == "__main__":
    main()
