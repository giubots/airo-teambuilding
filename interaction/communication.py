import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from gc import callbacks
from threading import Condition, Lock, Thread
from tkinter import N
from typing import Callable, Generator

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
    pending: dict[str, dict | None]
    pending_cond: Condition
    clients: list[ServerConnection]
    clients_lock: Lock
    callbacks: dict[str, list[Callable]]
    callbacks_lock: Lock
    ws_th: Thread

    def __init__(self, port: int = 8765) -> None:
        self.logger = logging.getLogger(__name__)
        self.pending = defaultdict(lambda: None)
        self.pending_cond = Condition()
        self.clients = []
        self.clients_lock = Lock()
        self.callbacks = defaultdict(list)
        self.callbacks_lock = Lock()
        server = serve(self._handler, "0.0.0.0", port)
        self.ws_th = Thread(target=server.serve_forever, daemon=True)
        self.ws_th.start()

    def _handler(self, websocket) -> None:
        self.logger.info("Client connected")
        with self.clients_lock:
            self.clients.append(websocket)
        try:
            for message in websocket:
                data = json.loads(message)
                self._send_to_all_clients(message)
                with self.callbacks_lock:
                    for callback in self.callbacks[data["type"]]:
                        callback(data["payload"])
                with self.pending_cond:
                    self.pending[data["type"]] = data["payload"]
                    self.pending_cond.notify_all()
        except ConnectionClosed:
            self.logger.info("Client disconnected from _handler")
        finally:
            with self.clients_lock:
                self.clients.remove(websocket)

    def _send_to_all_clients(self, message: str) -> None:
        to_remove = []
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.send(message)
                except ConnectionClosed:
                    self.logger.info("Client disconnected from _send_to_all_clients")
                    to_remove.append(client)
            for client in to_remove:
                self.clients.remove(client)

    def _send(self, type: str, payload: dict) -> None:
        message = json.dumps({"stamp": time.time(), "type": type, "payload": payload})
        self._send_to_all_clients(message)

    def _wait_for(self, m_type: str) -> dict:
        while True:
            with self.pending_cond:
                if self.pending[m_type] is not None:
                    return self.pending[m_type]  # type: ignore
                self.pending_cond.wait()

    def register_callback(self, m_type: str, callback: Callable) -> None:
        with self.callbacks_lock:
            self.callbacks[m_type].append(callback)

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
