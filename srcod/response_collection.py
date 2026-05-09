from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from .answer_parser import AnswerParser
from .loaders import QAItem
from .prompt_builder import PromptBuilder
from .utils import ensure_dir, safe_name

LOGGER = logging.getLogger(__name__)


class ChatClient(Protocol):
    def chat(self, model: str, prompt: str, image_paths: list[str] | None = None) -> tuple[str | None, str | None]:
        ...


def collect_responses(
    items: list[QAItem],
    models: list[str],
    dataset: str,
    client: ChatClient,
    responses_path: str | Path,
    per_model_dir: str | Path,
    overwrite: bool = False,
    save_every: int = 20,
) -> pd.DataFrame:
    responses_path = Path(responses_path)
    per_model_dir = Path(per_model_dir)
    ensure_dir(responses_path.parent)
    ensure_dir(per_model_dir)
    if overwrite and responses_path.exists():
        responses_path.unlink()

    parser = AnswerParser()
    prompt_builder = PromptBuilder()
    frames: list[pd.DataFrame] = []
    total = len(items) * len(models) * 4
    done = 0

    for model_id in models:
        model_path = per_model_dir / safe_name(model_id) / "responses.parquet"
        ensure_dir(model_path.parent)
        if overwrite and model_path.exists():
            model_path.unlink()
        existing = pd.read_parquet(model_path) if model_path.exists() else pd.DataFrame()
        finished_keys = _existing_keys(existing)
        records: list[dict[str, object]] = []

        for item in items:
            valid_letters = [chr(ord("A") + i) for i in range(len(item.choices))]
            for packet in prompt_builder.build_all_modes(item):
                key = (str(model_id), str(item.item_id), str(packet.mode))
                if key in finished_keys:
                    done += 1
                    continue
                raw_response, call_error = client.chat(
                    model=model_id,
                    prompt=packet.prompt,
                    image_paths=item.image_paths if packet.use_images else [],
                )
                pred_answer, parse_error = (None, None)
                if raw_response is not None:
                    pred_answer, parse_error = parser.parse(raw_response, valid_letters=valid_letters)
                correct_opt = parser.is_correct(pred_answer, item.answer)
                records.append(
                    {
                        "dataset": dataset,
                        "model_id": model_id,
                        "item_id": item.item_id,
                        "mode": packet.mode,
                        "s_image": packet.s_image,
                        "s_text": packet.s_text,
                        "gold_answer": item.answer,
                        "pred_answer": pred_answer,
                        "correct": 0 if correct_opt is None else int(correct_opt),
                        "raw_response": raw_response,
                        "error_message": call_error or parse_error,
                    }
                )
                done += 1
                if done % 10 == 0 or done == total:
                    LOGGER.info("Response progress %s/%s", done, total)
                if save_every > 0 and len(records) >= save_every:
                    existing = _flush(model_path, existing, records)
                    finished_keys = _existing_keys(existing)
        if records:
            existing = _flush(model_path, existing, records)
        if not existing.empty:
            frames.append(existing)

    if not frames:
        raise ValueError("No responses were collected.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["model_id", "item_id", "mode"], keep="last")
    merged.to_parquet(responses_path, index=False)
    LOGGER.info("Saved merged responses: %s rows -> %s", len(merged), responses_path)
    return merged


def _existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if df.empty:
        return set()
    return set(zip(df["model_id"].astype(str), df["item_id"].astype(str), df["mode"].astype(str)))


def _flush(path: Path, existing: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame.from_records(records)
    records.clear()
    if not existing.empty:
        merged = pd.concat([existing, new_df], ignore_index=True)
    elif path.exists():
        merged = pd.concat([pd.read_parquet(path), new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.drop_duplicates(subset=["model_id", "item_id", "mode"], keep="last")
    merged.to_parquet(path, index=False)
    return merged
