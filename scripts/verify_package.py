from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = PACKAGE_ROOT / "data" / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not (PACKAGE_ROOT / "data" / "lsmi_priors").exists(), "data/lsmi_priors must not be packaged"
    assert not (PACKAGE_ROOT / "response_matrix").exists(), "response_matrix/ directory must not be packaged"

    reports = {}
    for dataset, meta in manifest["datasets"].items():
        items_path = PACKAGE_ROOT / meta["items_file"]
        labels_path = PACKAGE_ROOT / meta["labels_file"]
        if not items_path.exists():
            raise FileNotFoundError(items_path)
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        items = pd.read_parquet(items_path)
        labels = pd.read_csv(labels_path)
        item_ids = set(items["item_id"].astype(str))
        if list(labels.columns) != ["item_id", "source_type_label"]:
            raise AssertionError(f"{dataset}: labels.csv must contain exactly item_id,source_type_label")
        label_col = "item_id"
        label_ids = set(labels[label_col].astype(str))
        overlap = item_ids & label_ids
        if len(overlap) != len(items):
            raise AssertionError(f"{dataset}: label overlap mismatch items={len(items)} overlap={len(overlap)}")
        missing_images = _count_missing_images(dataset, items)
        if missing_images:
            raise AssertionError(f"{dataset}: missing image files={missing_images}")
        reports[dataset] = {
            "items": len(items),
            "labels": len(labels),
            "label_overlap": len(overlap),
        }

    _scan_for_forbidden_secrets()
    print(json.dumps({"status": "ok", "datasets": reports}, ensure_ascii=False, indent=2))


def _count_missing_images(dataset: str, items: pd.DataFrame) -> int:
    if dataset == "MMMU":
        image_cols = [c for c in items.columns if c.startswith("image_")]
        has_image = items[image_cols].notna().any(axis=1) if image_cols else pd.Series(False, index=items.index)
        return int((~has_image).sum())
    dataset_dir = PACKAGE_ROOT / "data" / dataset
    return sum(0 if (dataset_dir / str(path)).exists() else 1 for path in items["image_path"].astype(str))


def _scan_for_forbidden_secrets() -> None:
    forbidden_patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    ]
    text_suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv"}
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                raise AssertionError(f"Forbidden endpoint/key-like token found in {path}")


if __name__ == "__main__":
    main()
