from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class QAItem:
    item_id: str
    question: str
    choices: list[str]
    answer: str
    image_paths: list[str]
    category: str | None = None
    split: str | None = None
    raw: dict[str, Any] | None = None


DATASET_DIRS = {
    "mmmu": "MMMU",
    "scienceqa": "ScienceQA",
    "seedbench": "SEED-Bench",
    "seed-bench": "SEED-Bench",
}


def normalize_dataset_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_DIRS:
        raise ValueError(f"Unsupported dataset={dataset}. Expected one of: MMMU, ScienceQA, SEED-Bench")
    return key


def dataset_dir_name(dataset: str) -> str:
    return DATASET_DIRS[normalize_dataset_name(dataset)]


def load_items(
    dataset: str,
    data_root: str | Path,
    image_cache_dir: str | Path,
    max_items: int | None = None,
) -> tuple[list[QAItem], dict[str, Any]]:
    dataset_key = normalize_dataset_name(dataset)
    root = Path(data_root)
    if root.name != dataset_dir_name(dataset_key):
        root = root / dataset_dir_name(dataset_key)
    items_path = root / "items.parquet"
    if not items_path.exists():
        raise FileNotFoundError(f"items.parquet not found: {items_path}")
    df = pd.read_parquet(items_path)

    if dataset_key == "mmmu":
        items = _load_mmmu(df, image_cache_dir=Path(image_cache_dir), max_items=max_items)
    elif dataset_key == "scienceqa":
        items = _load_path_based(df, dataset_dir=root, max_items=max_items)
    else:
        items = _load_path_based(df, dataset_dir=root, max_items=max_items)

    stats = {
        "dataset": dataset_dir_name(dataset_key),
        "items_path": str(items_path),
        "total_rows": int(len(df)),
        "loaded_items": int(len(items)),
    }
    LOGGER.info("Loaded %s items from %s", len(items), items_path)
    return items, stats


def _load_mmmu(df: pd.DataFrame, image_cache_dir: Path, max_items: int | None) -> list[QAItem]:
    ensure_dir(image_cache_dir)
    out: list[QAItem] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        item_id = str(row_dict.get("item_id") or "").strip()
        question = str(row_dict.get("question") or "").strip()
        choices = _parse_choices(row_dict.get("options"))
        answer = _normalize_answer(row_dict.get("answer"))
        if not item_id or not question or not choices or not answer:
            continue
        image_paths = _extract_mmmu_images(row_dict, image_cache_dir=image_cache_dir)
        if not image_paths:
            continue
        out.append(
            QAItem(
                item_id=item_id,
                question=question,
                choices=choices,
                answer=answer,
                image_paths=image_paths,
                category=_as_optional_str(row_dict.get("category") or row_dict.get("source_subject")),
                split=_as_optional_str(row_dict.get("split") or row_dict.get("source_split")),
                raw=row_dict,
            )
        )
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _load_path_based(df: pd.DataFrame, dataset_dir: Path, max_items: int | None) -> list[QAItem]:
    out: list[QAItem] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        item_id = str(row_dict.get("item_id") or "").strip()
        question = str(row_dict.get("question") or "").strip()
        choices = _parse_choices(row_dict.get("choices"))
        answer = _normalize_answer(row_dict.get("gold_answer") or row_dict.get("answer"))
        raw_image_path = str(row_dict.get("image_path") or "").strip()
        image_path = (dataset_dir / raw_image_path).resolve()
        if not item_id or not question or not choices or not answer or not image_path.exists():
            continue
        out.append(
            QAItem(
                item_id=item_id,
                question=question,
                choices=choices,
                answer=answer,
                image_paths=[str(image_path)],
                category=_as_optional_str(row_dict.get("category")),
                split=_as_optional_str(row_dict.get("split")),
                raw=row_dict,
            )
        )
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _parse_choices(raw: Any) -> list[str] | None:
    value = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = ast.literal_eval(text)
        except Exception:
            return None
    if not isinstance(value, list):
        return None
    choices = [str(x).strip() for x in value]
    if len(choices) < 2 or any(not x for x in choices):
        return None
    return choices


def _normalize_answer(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    return None


def _extract_mmmu_images(row: dict[str, Any], image_cache_dir: Path) -> list[str]:
    item_id = str(row.get("item_id") or row.get("id") or "item")
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item_id)
    paths: list[str] = []
    for idx in range(1, 8):
        value = row.get(f"image_{idx}")
        if value is None:
            continue
        image_bytes = None
        path_value = None
        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            path_value = value.get("path")
        elif isinstance(value, str):
            path_value = value
        if image_bytes:
            ext = _detect_image_ext(image_bytes)
            out_path = image_cache_dir / f"{safe_id}_{idx}{ext}"
            if not out_path.exists():
                out_path.write_bytes(image_bytes)
            paths.append(str(out_path.resolve()))
        elif path_value:
            p = Path(str(path_value))
            if p.exists():
                paths.append(str(p.resolve()))
    return paths


def _detect_image_ext(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:32]:
        return ".webp"
    return ".img"


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
