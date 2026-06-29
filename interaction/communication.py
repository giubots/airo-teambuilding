import json
import logging
import time
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Generator

from websockets import ConnectionClosed
from websockets.sync.server import ServerConnection, serve


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
    logger: logging.Logger
    new_pending: Condition
    pending: list[dict]
    clients: list[ServerConnection]
    ws_th: Thread

    def __init__(self, port: int = 8765) -> None:
        self.logger = logging.getLogger(__name__)
        self.new_pending = Condition()
        self.pending = []
        server = serve(self._handler, "localhost", port)
        self.ws_th = Thread(target=server.serve_forever, daemon=True)
        self.ws_th.start()

    def _send_to_all_clients(self, message: str) -> None:
        for client in self.clients:
            try:
                client.send(message)
            except ConnectionClosed:
                self.clients.remove(client)

    def _handler(self, websocket) -> None:
        self.clients.append(websocket)  # FIXME: multithreading will break
        for message in websocket:
            data = json.loads(message)
            self._send_to_all_clients(message)
            with self.new_pending:
                self.pending.append(data)
                self.new_pending.notify_all()

    def _wait_for(self, m_type: str) -> dict:
        while True:
            with self.new_pending:
                for i, data in enumerate(self.pending):
                    if data["type"] == m_type:
                        return self.pending.pop(i)["payload"]
                self.new_pending.wait()

    def _send(self, type: str, payload: dict) -> None:
        self._send_to_all_clients(
            json.dumps(
                {
                    "stamp": time.time(),
                    "type": type,
                    "payload": payload,
                }
            )
        )

    def look_async(self) -> None:
        self._send("look", {})

    def wait_done_look(self) -> DoneLook:
        return DoneLook(**self._wait_for("done-look"))

    def look(self) -> DoneLook:
        self.look_async()
        return self.wait_done_look()

    def find_async(self, noun: str, adjectives: list[str]) -> None:
        self._send("find", {"noun": noun, "adj": adjectives})

    def wait_done_find(self) -> DoneFind:
        return DoneFind(**self._wait_for("done-find"))

    def find(self, noun: str, adjectives: list[str]) -> DoneFind:
        self.find_async(noun, adjectives)
        return self.wait_done_find()

    def deliver_async(self, payload: object) -> None:
        self._send("deliver", {"payload": payload})

    def wait_done_deliver(self) -> DoneDeliver:
        return DoneDeliver(**self._wait_for("done-deliver"))

    def deliver(self, payload: object) -> DoneDeliver:
        self.deliver_async(payload)
        return self.wait_done_deliver()

    def stop_async(self) -> None:
        self._send("stop", {})

    def wait_done_stop(self) -> DoneStop:
        return DoneStop(**self._wait_for("done-stop"))

    def stop(self) -> DoneStop:
        self.stop_async()
        return self.wait_done_stop()

    def get_feedback(self) -> Generator[Feedback, None, None]:
        while True:
            yield Feedback(**self._wait_for("feedback"))
