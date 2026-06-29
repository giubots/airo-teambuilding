"""A scripted 'backpack' playbook for the interaction backend's Reachy Mini.

Mirror of ``reachy/core/playbook.py`` adapted to the :class:`ReachyMiniRobot`
class. Strict turn-taking: the robot is EITHER listening/talking OR thinking,
never both at once. The flow is:

  1. Greet: 'Hi, nice to see you! What do you want me to grab from your backpack?'
  2. Listen, then ask the LLM for the object as ``{noun, adjective}`` JSON
     (at most one relevant adjective), saved to a JSON file and securely copied
     to a remote server over SSH (SCP) for the picker to use.
  3. Say 'Okay, let me search for <words>.'
  4. Enter thinking/dance mode while the robot 'picks' the item from the backpack.
  5. When the operator ends thinking mode, say 'Happy to help!' and loop.

Thinking is driven through :meth:`ReachyMiniRobot.set_thinking`, so the head and
music stay with the robot's follow loop and the two modes never overlap.
"""

import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# Find the SCP/OpenAI settings in the cwd and the repo root so they are picked
# up regardless of which directory the demo is launched from.
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EXTRACT_MODEL = "gpt-4o-mini"
REQUEST_PATH = os.path.join(os.getcwd(), "backpack_request.json")

_openai = None
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _openai = OpenAI()
    except Exception:
        _openai = None

# Lionel is easily mis-heard, so accept a few close transcriptions of the name.
_WAKE_HINTS = ("lionel", "leonel", "lional", "lyonel", "lionell", "lion")


def is_wake(text: str) -> bool:
    """True when the transcript sounds like the 'Hi Lionel!' wake phrase."""
    t = (text or "").lower()
    return any(h in t for h in _WAKE_HINTS)


def item_words(item: dict) -> str:
    """Human phrase for the request, e.g. 'red bottle' or just 'bottle'."""
    adj = (item.get("adjective") or "").strip()
    noun = (item.get("noun") or "").strip()
    return f"{adj} {noun}".strip() if adj else noun


def extract_item(text: str) -> dict:
    """Ask the LLM for the object as {noun, adjective(<=1, optional)} JSON."""
    words = text.strip().split()
    fallback = {"noun": words[-1].lower() if words else "", "adjective": None}
    if _openai is None or not text.strip():
        return fallback
    system = (
        "You extract which single object a person wants taken from their "
        "backpack. Return strict JSON with exactly these keys: "
        "'noun' (string: the one object, singular, lowercase) and "
        "'adjective' (a single relevant descriptive adjective such as a colour, "
        "size or material, or null if none is clearly stated). "
        "Include at most one adjective, and only if it is clearly relevant."
    )
    try:
        r = _openai.chat.completions.create(
            model=EXTRACT_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text}])
        data = json.loads(r.choices[0].message.content or "{}")
        noun = str(data.get("noun") or "").strip().lower()
        adj = data.get("adjective")
        adj = str(adj).strip().lower() if adj else None
        return {"noun": noun, "adjective": adj or None} if noun else fallback
    except Exception:
        return fallback


def save_request(item: dict, path: str = REQUEST_PATH) -> str:
    """Write the structured request to a JSON file; return its path."""
    payload = {
        "noun": item.get("noun"),
        "adjective": item.get("adjective"),
        "words": item_words(item),
        "timestamp": time.time(),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass
    return path


def _scp_config() -> dict:
    """Read the SCP upload settings from the environment (.env)."""
    enabled = (os.getenv("BACKPACK_SCP_ENABLED", "false").strip().lower()
               in ("1", "true", "yes", "on"))
    return {
        "enabled": enabled,
        "host": os.getenv("BACKPACK_SCP_HOST", "").strip(),
        "port": int(os.getenv("BACKPACK_SCP_PORT", "22") or 22),
        "user": os.getenv("BACKPACK_SCP_USER", "").strip(),
        "password": os.getenv("BACKPACK_SCP_PASSWORD") or None,
        "key_file": os.getenv("BACKPACK_SCP_KEY", "").strip() or None,
        "remote_path": os.getenv("BACKPACK_SCP_REMOTE_PATH", "").strip(),
        "timeout": float(os.getenv("BACKPACK_SCP_TIMEOUT", "10") or 10),
    }


def _ensure_remote_dir(ssh, remote_path: str) -> None:
    """Best-effort 'mkdir -p' of the directory that will hold remote_path.

    SCP errors out when the destination directory is missing, so create the
    parent chain over SFTP first. Never raises - it is purely a convenience.
    """
    import posixpath
    directory = (remote_path if remote_path.endswith("/")
                 else posixpath.dirname(remote_path)).rstrip("/")
    if not directory:
        return
    sftp = ssh.open_sftp()
    try:
        cur = "/" if directory.startswith("/") else ""
        for part in directory.split("/"):
            if not part:
                continue
            cur = posixpath.join(cur, part) if cur else part
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass
    finally:
        try:
            sftp.close()
        except Exception:
            pass


def upload_request(path: str = REQUEST_PATH, cfg: dict = None) -> bool:
    """Securely copy the saved request to the remote server over SSH (SCP).

    Connection details are read from the environment (see the ``.env`` block).
    The transfer runs over an encrypted SSH channel. Key-based auth is used when
    ``BACKPACK_SCP_KEY`` is set, otherwise the password. This never raises -
    problems are printed and ``False`` is returned so the playbook keeps going.
    """
    cfg = cfg or _scp_config()
    if not cfg["enabled"]:
        return False
    missing = [k for k in ("host", "user", "remote_path") if not cfg[k]]
    if missing:
        print("[scp] upload skipped; set "
              + ", ".join("BACKPACK_SCP_" + m.upper() for m in missing)
              + " in .env")
        return False
    if not os.path.exists(path):
        print(f"[scp] upload skipped; {path} not found")
        return False
    try:
        import paramiko
        from scp import SCPClient
    except Exception as e:
        print(f"[scp] needs 'paramiko' and 'scp' (pip install paramiko scp): {e}")
        return False

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect = {
            "hostname": cfg["host"], "port": cfg["port"],
            "username": cfg["user"], "timeout": cfg["timeout"],
            "allow_agent": False, "look_for_keys": False,
        }
        if cfg["key_file"]:
            connect["key_filename"] = cfg["key_file"]
        else:
            connect["password"] = cfg["password"]
        ssh.connect(**connect)
        _ensure_remote_dir(ssh, cfg["remote_path"])
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(path, cfg["remote_path"])
        print(f"[scp] uploaded {os.path.basename(path)} to "
              f"{cfg['user']}@{cfg['host']}:{cfg['remote_path']}")
        return True
    except Exception as e:
        print(f"[scp] upload failed: {e}")
        return False
    finally:
        try:
            ssh.close()
        except Exception:
            pass


def upload_request_async(path: str = REQUEST_PATH) -> None:
    """Upload in a daemon thread so the robot never blocks on the network."""
    if not _scp_config()["enabled"]:
        return
    threading.Thread(target=upload_request, args=(path,), daemon=True).start()


def run_playbook(robot, stop: "threading.Event", listen_secs: float = 4.0) -> None:
    """Run the backpack playbook until ``stop`` is set.

    ``robot.thinking`` is the mutually-exclusive mode flag: this loop only
    listens/talks while it is False, and enters thinking via
    ``robot.set_thinking(True)`` (the operator turns it back off to signal
    'done picking').
    """
    print("Playbook ready. Lionel will greet and ask what to fetch.")
    while not stop.is_set():
        # --- LISTEN/TALK mode: never while thinking -------------------------
        if robot.thinking:
            time.sleep(0.15)
            continue

        # 1. Greet and ask what to fetch.
        robot.say("Hi, nice to see you! What do you want me to grab "
                  "from your backpack?", block=True)

        # 2. Listen for the item (a couple of tries).
        item_text = ""
        for _ in range(2):
            if stop.is_set():
                break
            item_text = robot.listen(listen_secs + 2.0)
            if item_text.strip():
                break
            robot.say("Sorry, I didn't catch that. What should I grab?",
                      block=True)
        if stop.is_set():
            break
        if not item_text.strip():
            robot.say("No worries, let's try again later!", block=True)
            continue
        print(f"You: {item_text}")

        # 3. Extract the structured request, save it and upload it for the picker.
        item = extract_item(item_text)
        path = save_request(item)
        upload_request_async(path)
        words = item_words(item) or "that"
        print(f"Backpack request -> {json.dumps(item)}  (saved: {path})")

        # 4. Acknowledge what we're looking for.
        robot.say(f"Okay, let me search for {words}.", block=True)

        # 5. --- THINKING mode: pick the item; no listening/talking here ------
        robot.set_thinking(True)

        # 6. Wait for the operator to end thinking mode (the 'picking' is done).
        while robot.thinking and not stop.is_set():
            time.sleep(0.1)
        if stop.is_set():
            break

        # 7. Back to talking mode.
        robot.say("Happy to help!", block=True)


def start_playbook(robot, listen_secs: float = 4.0) -> "threading.Event":
    """Spawn run_playbook in a daemon thread; returns the stop Event."""
    stop = threading.Event()
    threading.Thread(target=run_playbook, args=(robot, stop, listen_secs),
                     daemon=True).start()
    return stop
