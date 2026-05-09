from __future__ import annotations

import re
from typing import Iterable


class AnswerParser:
    def parse(self, text: str, valid_letters: Iterable[str]) -> tuple[str | None, str | None]:
        valid = {str(x).upper() for x in valid_letters}
        if not valid:
            return None, "empty_valid_letters"
        raw = "" if text is None else str(text).strip()
        if not raw:
            return None, "empty_response"

        first = raw.strip().upper()[:1]
        if first in valid:
            return first, None

        patterns = [
            r"\b(?:ANSWER|OPTION|CHOICE)\s*[:：]?\s*([A-Z])\b",
            r"\(([A-Z])\)",
            r"\b([A-Z])\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
                letter = match.group(1).upper()
                if letter in valid:
                    return letter, None
        return None, f"could_not_parse:{raw[:120]}"

    @staticmethod
    def is_correct(pred_answer: str | None, gold_answer: str | None) -> int | None:
        if pred_answer is None or gold_answer is None:
            return None
        pred = str(pred_answer).strip().upper()
        gold = str(gold_answer).strip().upper()
        if not pred or not gold:
            return None
        return int(pred == gold)
