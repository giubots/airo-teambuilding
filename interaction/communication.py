import json
import logging
import time
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Any, Generator

from websockets.sync.server import serve


@dataclass
class DoneLook:
    success: bool
    img: str


@dataclass
class DoneFind:
    success: bool
    message: str
    payload: object


@dataclass
class DoneDeliver:
    success: bool
    message: str


@dataclass
class DoneStop:
    success: bool
    message: str


@dataclass
class Feedback:
    message: str


class Communication:
    """Starts a websocket server and handles the communication.

    This sends a request and waits on a threading event for a response of a certain type. The response is then returned to the caller.
    """

    logger: logging.Logger
    event: Condition
    pending: list[dict]
    server: Any
    ws_th: Thread

    def __init__(self, port: int = 8765) -> None:
        """Starts a websocket server on the given port."""
        self.logger = logging.getLogger(__name__)
        self.event = Condition()
        self.pending = []
        self.server = serve(self._handler, "localhost", port)
        self.ws_th = Thread(target=self.server.serve_forever, daemon=True)
        self.ws_th.start()

    def _handler(self, websocket) -> None:
        """Handles incoming messages from the client."""
        for message in websocket:
            data = json.loads(message)
            with self.event:
                self.pending.append(data)
                self.event.notify_all()

    def _look_for(self, type: str) -> dict:
        """Looks for a response of a certain type in the queue."""
        while True:
            with self.event:
                for i, data in enumerate(self.pending):
                    if data["type"] == type:
                        return self.pending.pop(i)
                self.event.wait()

    def _send(self, type: str, payload: dict) -> None:
        """Sends a request to the client."""
        self.server.send(
            json.dumps(
                {
                    "stamp": time.time(),
                    "type": type,
                    "payload": payload,
                }
            )
        )

    def look(self) -> DoneLook:
        """Sends a look request to the client and waits for the response."""
        self._send("look", {})
        return DoneLook(**self._look_for("done-look"))

    def find(self, noun: str, adjectives: list[str]) -> DoneFind:
        """Sends a find request to the client and waits for the response."""
        self._send("find", {"noun": noun, "adj": adjectives})
        return DoneFind(**self._look_for("done-find"))

    def deliver(self, payload: object) -> DoneDeliver:
        """Sends a deliver request to the client and waits for the response."""
        self._send("deliver", {"payload": payload})
        return DoneDeliver(**self._look_for("done-deliver"))

    def stop(self) -> DoneStop:
        """Sends a stop request to the client and waits for the response."""
        self._send("stop", {})
        return DoneStop(**self._look_for("done-stop"))

    def get_feedback(self) -> Generator[Feedback, None, None]:
        """Blocks until feedback is received from the client and yields it."""
        while True:
            yield Feedback(**self._look_for("feedback"))
