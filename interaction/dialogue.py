import logging
from collections.abc import Generator
from concurrent.futures import Future

from vision_enabled_dialogue.conversation_history import ConversationHistory
from vision_enabled_dialogue.conversation_history.strategies import SummariseHeavy
from vision_enabled_dialogue.conversation_history.summarisers import HidingSummariser
from vision_enabled_dialogue.dialogue_manager import DialogueManager
from vision_enabled_dialogue.llm import GPT
from vision_enabled_dialogue.response_gen import VLMResponseGenerator


class DemoDialogue:
    logger: logging.Logger
    generator: VLMResponseGenerator
    dm: DialogueManager

    def __init__(self, instructions: str, vlm: GPT):
        self.logger = logging.getLogger(__name__)
        history = ConversationHistory(
            strategy_factories=[
                lambda: SummariseHeavy(
                    max_heavy=5,
                    min_heavy=1,
                    summariser_factories=[HidingSummariser],
                ),
            ]
        )
        # history.on_update = self._on_conversation_update
        # history.on_full_update = self._on_full_conversation_update
        self.generator = VLMResponseGenerator(vlm=vlm)
        self.generator.INSTRUCTIONS = instructions
        self.dm = DialogueManager(
            self.speak,
            history,
            self.generator,
            True,
        )

    def on_steering(self, msg: str, stamp: float):
        self.dm.add_steering(msg, stamp)

    def add_thought(self, msg: str, stamp: float):
        self.dm.add_thought(msg, stamp)

    def force_turn(self, steering: str):
        self.dm.generate_response(steering=steering)

    def on_speech(self, msg: str, stamp: float):
        self.dm.add_turn(msg, stamp)
        self.dm.generate_response()

    def on_image(self, msg: str, stamp: float):
        self.dm.add_frame(msg, stamp)

    def speak(self, speech_g: Generator[str]) -> Future[str]:
        string_f = Future[str]()
        spoken = []

        for text in speech_g:
            print(f"Speaking: {text}")
            spoken.append(text)
        string_f.set_result(" ".join(spoken))
        return string_f
