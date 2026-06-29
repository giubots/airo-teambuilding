# interaction

Backend that drives the robot during a demo: dialogue, websocket communication,
and a **Reachy Mini** baked in.

- `robot.py` — `ReachyMiniRobot`: greet, antenna "happy", `look()` (camera), and
  continuous forefront-face following (`FaceTracker`). Used by `main.py`.
- `robot_demo.py` — try it alone: `python -m interaction.robot_demo`
- `communication.py` — websocket command bridge · `dialogue.py` — LLM dialogue

## Robot
1. Plug in the Reachy Mini and start the daemon: `reachy-mini-daemon` (`--sim` for sim).
2. `pip install -r requirements.txt` then run `python main.py`.

`main.py` runs without a robot too (it logs a warning and skips it).
