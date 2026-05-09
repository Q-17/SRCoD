from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from srcod.train import evaluate_item_labels, train_srcod
from srcod.utils import ensure_dir, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SRCoD.")
    p.add_argument("--responses-path", required=True, type=str)
    p.add_argument("--lsmi-metrics-path", required=True, type=str)
    p.add_argument("--output-dir", type=str, default=str(PACKAGE_ROOT / "outputs" / "srcod"))
    p.add_argument("--labels-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lambda-profile-kl", type=float, default=0.1)
    p.add_argument("--tau-lsmi-prior", type=float, default=1.0)
    p.add_argument("--z-clip-quantile", type=float, default=0.995)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    responses_path = Path(args.responses_path)
    lsmi_metrics_path = Path(args.lsmi_metrics_path)
    if not responses_path.exists():
        raise FileNotFoundError(f"responses-path not found: {responses_path}")
    if not lsmi_metrics_path.exists():
        raise FileNotFoundError(
            f"lsmi-metrics-path not found: {lsmi_metrics_path}. "
            "This submission package does not include LSMI priors; pass the external LSMI metrics CSV explicitly."
        )

    responses = pd.read_parquet(responses_path)
    lsmi_metrics = pd.read_csv(lsmi_metrics_path)
    result = train_srcod(
        responses,
        lsmi_metrics,
        seed=args.seed,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        lambda_profile_kl=args.lambda_profile_kl,
        tau_lsmi_prior=args.tau_lsmi_prior,
        z_clip_quantile=args.z_clip_quantile,
    )

    out_dir = ensure_dir(args.output_dir)
    result.item_scores.to_csv(out_dir / "item_scores.csv", index=False)
    result.model_scores.to_csv(out_dir / "model_scores.csv", index=False)
    pd.DataFrame.from_records(result.history).to_csv(out_dir / "train_history.csv", index=False)
    summary = dict(result.summary)
    if args.labels_path:
        labels_path = Path(args.labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"labels-path not found: {labels_path}")
        summary["label_eval"] = evaluate_item_labels(result.item_scores, pd.read_csv(labels_path))
    save_json(summary, out_dir / "summary.json")
    print({"output_dir": str(out_dir), "summary": summary})


if __name__ == "__main__":
    main()
