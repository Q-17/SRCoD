from __future__ import annotations

from dataclasses import dataclass

from .loaders import QAItem


MODE_TO_FLAGS = {
    "it": (1, 1),
    "i": (1, 0),
    "t": (0, 1),
    "none": (0, 0),
}


@dataclass(frozen=True)
class PromptPacket:
    mode: str
    prompt: str
    s_image: int
    s_text: int
    use_images: bool


class PromptBuilder:
    def build_all_modes(self, item: QAItem) -> list[PromptPacket]:
        return [self.build(item, mode) for mode in ["it", "i", "t", "none"]]

    def build(self, item: QAItem, mode: str) -> PromptPacket:
        if mode not in MODE_TO_FLAGS:
            raise ValueError(f"Unknown mode: {mode}")

        s_image, s_text = MODE_TO_FLAGS[mode]
        options_block = self._format_options(item.choices)
        valid_letters = ", ".join(chr(ord("A") + i) for i in range(len(item.choices)))
        instruction = (
            "You are solving a multiple-choice question. "
            f"Output exactly one uppercase letter from [{valid_letters}] and nothing else."
        )

        if mode == "it":
            body = (
                "Mode: image + text.\n"
                "Use both the provided image(s) and question text.\n"
                f"Question:\n{item.question}\n\n"
                f"Options:\n{options_block}\n"
            )
            use_images = True
        elif mode == "i":
            body = (
                "Mode: image only.\n"
                "Question text is hidden. Use only the provided image(s) and options.\n"
                f"Options:\n{options_block}\n"
            )
            use_images = True
        elif mode == "t":
            body = (
                "Mode: text only.\n"
                "No image is available. Use only question text and options.\n"
                f"Question:\n{item.question}\n\n"
                f"Options:\n{options_block}\n"
            )
            use_images = False
        else:
            body = (
                "Mode: none.\n"
                "No image and no question text are available.\n"
                "Choose the most plausible answer only from the options below. Guess if needed.\n"
                f"Options:\n{options_block}\n"
            )
            use_images = False

        return PromptPacket(
            mode=mode,
            prompt=f"{instruction}\n\n{body}\nAnswer:",
            s_image=s_image,
            s_text=s_text,
            use_images=use_images,
        )

    @staticmethod
    def _format_options(choices: list[str]) -> str:
        return "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
