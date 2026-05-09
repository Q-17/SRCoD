from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from srcod.clients import OllamaClient, OpenAICompatibleClient
from srcod.loaders import dataset_dir_name, load_items
from srcod.response_collection import collect_responses
from srcod.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect multimodal response matrices for SRCoD.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", choices=["MMMU", "ScienceQA", "SEED-Bench"], default=None)
    p.add_argument("--backend", choices=["ollama", "openai"], default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--responses-path", type=str, default=None)
    p.add_argument("--per-model-dir", type=str, default=None)
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--retry-backoff", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--ollama-base-url", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL.")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--api-key-env", type=str, default=None)
    p.add_argument("--log-level", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    setup_logging(_pick(args.log_level, cfg, "log_level", "INFO"))

    dataset = _pick(args.dataset, cfg, "dataset", None)
    if not dataset:
        raise ValueError("Missing --dataset or config.dataset")
    backend = _pick(args.backend, cfg, "backend", "ollama")
    models = args.models if args.models is not None else cfg.get("models", [])
    if not models:
        raise ValueError("Missing --models or config.models")

    data_root = Path(_pick(args.data_root, cfg, "data_root", str(PACKAGE_ROOT / "data")))
    output_dir = Path(_pick(args.output_dir, cfg, "output_dir", str(PACKAGE_ROOT / "outputs" / f"{dataset_dir_name(dataset)}_responses")))
    responses_path = Path(_pick(args.responses_path, cfg, "responses_path", str(output_dir / "responses.parquet")))
    per_model_dir = Path(_pick(args.per_model_dir, cfg, "per_model_dir", str(output_dir / "responses_by_model")))
    image_cache_dir = output_dir / "image_cache"

    items, loader_stats = load_items(
        dataset=dataset,
        data_root=data_root,
        image_cache_dir=image_cache_dir,
        max_items=_pick(args.max_items, cfg, "max_items", None),
    )
    if not items:
        raise ValueError(f"No valid items loaded for dataset={dataset}")

    common_client_kwargs = {
        "timeout_sec": float(_pick(args.timeout, cfg, "timeout", 600.0)),
        "max_retries": int(_pick(args.max_retries, cfg, "max_retries", 2)),
        "retry_backoff_sec": float(_pick(args.retry_backoff, cfg, "retry_backoff", 2.0)),
        "temperature": float(_pick(args.temperature, cfg, "temperature", 0.0)),
    }
    if backend == "ollama":
        client = OllamaClient(
            base_url=_pick(args.ollama_base_url, cfg, "ollama_base_url", "http://127.0.0.1:11434"),
            **common_client_kwargs,
        )
    else:
        client = OpenAICompatibleClient(
            base_url=_pick(args.base_url, cfg, "base_url", "https://api.openai.com/v1"),
            api_key=_pick(args.api_key, cfg, "api_key", None),
            api_key_env=_pick(args.api_key_env, cfg, "api_key_env", "OPENAI_API_KEY"),
            **common_client_kwargs,
        )

    df = collect_responses(
        items=items,
        models=[str(m) for m in models],
        dataset=dataset_dir_name(dataset),
        client=client,
        responses_path=responses_path,
        per_model_dir=per_model_dir,
        overwrite=bool(args.overwrite or cfg.get("overwrite", False)),
        save_every=int(_pick(args.save_every, cfg, "save_every", 20)),
    )
    print(
        {
            "dataset": dataset,
            "backend": backend,
            "item_count": len(items),
            "num_response_rows": int(len(df)),
            "responses_path": str(responses_path),
            "loader_stats": loader_stats,
        }
    )


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick(value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    return value if value is not None else cfg.get(key, default)


if __name__ == "__main__":
    main()
