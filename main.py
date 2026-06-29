import threading
from time import time

from pydantic import BaseModel

from interaction.communication import Communication
from interaction.dialogue import DemoDialogue
from vision_enabled_dialogue.conversation_history.parts import Frame
from vision_enabled_dialogue.llm import GPT


class FindResponse(BaseModel):
    success: bool
    name: str
    descriptors: list[str]


class DemoInteraction:
    gpt: GPT
    dialogue: DemoDialogue
    comm: Communication

    def __init__(self, comm: Communication):
        instructions = (
            "You are a helpful assistant that can answer questions about the world."
        )
        tools_desc = [
            {
                "type": "function",
                "name": "try_to_fetch",
                "description": "Tries to fetch something from the backpack.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "What the user is asking for.",
                        }
                    },
                    "required": ["request"],
                    "additionalProperties": False,
                },
            },
        ]
        tools_impl = {"try_to_fetch": self.look_find_fetch}

        self.gpt = GPT(
            model="gpt-4.1-mini-2025-04-14",
            base_url="",
            api_key="",
            tools_desc=tools_desc,
            tools_impl=tools_impl,
        )
        self.dialogue = DemoDialogue(
            instructions=instructions,
            vlm=self.gpt,
        )
        self.comm = comm
        threading.Thread(target=self.communicate_feedback_loop, daemon=True).start()

    def look_find_fetch(self, request: str):
        print("Calling look_find_fetch with request:", request)
        self.comm.look_async()
        self.dialogue.add_thought("Looking in the backpack.", time())
        done_look = self.comm.wait_done_look()
        if not done_look.success:
            self.dialogue.add_thought("Failed to look in the backpack.", time())
            self.dialogue.force_turn("Report that the action failed")
            return

        self.dialogue.add_thought("Done looking in the backpack.", time())
        find_response = self.find_in_image(request, done_look.img)
        if not find_response.success:
            self.dialogue.add_thought("I looked cannot find a matching object.", time())
            self.dialogue.force_turn("Report that the action failed")
            return

        done_find = self.comm.find(find_response.name, find_response.descriptors)
        if not done_find.success:
            self.dialogue.add_thought(
                f"I cannot find a matching object: {done_find.message}",
                time(),
            )
            self.dialogue.force_turn("Report that the action failed")
            return

        self.dialogue.add_thought(
            f"I will fetch the object {find_response.name} with descriptors {find_response.descriptors}.",
            time(),
        )
        done_deliver = self.comm.deliver(payload=done_find.payload)
        if not done_deliver.success:
            self.dialogue.add_thought(
                f"Failed to deliver the object: {done_deliver.message}", time()
            )
            self.dialogue.force_turn("Report that the action failed")
            return

        self.dialogue.add_thought("Successfully delivered the object", time())

    def find_in_image(self, text: str, img_base64: str) -> FindResponse:
        i = f"Find the object that the user is asking for in the image. The user said: {text}."
        res, _ = self.gpt.formatted(i, [Frame(time(), img_base64)], FindResponse)
        return res

    def communicate_feedback_loop(self):
        for feedback in self.comm.get_feedback():
            self.dialogue.force_turn(
                f"Report feedback from the robot: {feedback.message}"
            )


def start_chat(comm: Communication):
    print("Starting chat")
    demo = DemoInteraction(comm)
    try:
        while True:
            user = input("User: ")
            demo.dialogue.dm.add_turn(user, time())
            demo.dialogue.dm.generate_response()
    except KeyboardInterrupt:
        demo.comm.stop_async()
        demo.comm.wait_done_stop()

def analyse():
    print("Analysing the image")

def relay_message():
    print("Relaying the message")


def main():
    comm = Communication()
    comm.register_callback("start", start_chat)
    comm.register_callback("done-look", analyse)
    comm.register_callback("done-find", relay_message)
    start_chat(comm)


if __name__ == "__main__":
    main()
