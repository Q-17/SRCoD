from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import set_random_seed

LOGGER = logging.getLogger(__name__)

LABELS = ["redundant", "image-unique", "text-unique", "synergistic"]
Z_NAMES = ["R", "U_img", "U_txt", "S"]
LSMI_FEATURES = ["R", "U1", "U2", "S"]


@dataclass
class SRCoDTrainResult:
    history: list[dict[str, float]]
    item_scores: pd.DataFrame
    model_scores: pd.DataFrame
    summary: dict[str, Any]


def train_srcod(
    responses: pd.DataFrame,
    lsmi_metrics: pd.DataFrame,
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    batch_size: int = 2048,
    device: str = "cpu",
    lr: float = 0.01,
    epochs: int = 300,
    lambda_profile_kl: float = 0.1,
    tau_lsmi_prior: float = 1.0,
    z_clip_quantile: float = 0.995,
) -> SRCoDTrainResult:
    required_cols = ["model_id", "item_id", "s_image", "s_text", "correct"]
    for col in required_cols:
        if col not in responses.columns:
            raise ValueError(f"responses missing required column: {col}")

    data = responses.copy()
    data = data[data["correct"].isin([0, 1])].reset_index(drop=True)
    item_z_df = _prepare_lsmi_item_features(lsmi_metrics)
    valid_item_ids = set(item_z_df["item_id"].astype(str))
    before_rows = len(data)
    data = data[data["item_id"].astype(str).isin(valid_item_ids)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No response rows remain after intersecting responses with LSMI metrics.")
    LOGGER.info("SRCoD data rows before=%s after_lsmi_join=%s", before_rows, len(data))

    set_random_seed(seed)
    torch.manual_seed(seed)
    model_ids = sorted(data["model_id"].astype(str).unique().tolist())
    item_ids = sorted(data["item_id"].astype(str).unique().tolist())
    model_to_idx = {model_id: idx for idx, model_id in enumerate(model_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    z_map = item_z_df.set_index("item_id")
    z_raw = z_map.loc[item_ids, [f"z_{name}" for name in Z_NAMES]].to_numpy(dtype=np.float64)
    z_raw = _clip_features(z_raw, z_clip_quantile=z_clip_quantile)
    z_mean = z_raw.mean(axis=0, keepdims=True)
    z_std = z_raw.std(axis=0, keepdims=True)
    z_std = np.where(z_std < 1e-8, 1.0, z_std)
    z_item_std = ((z_raw - z_mean) / z_std).astype(np.float32)
    q_lsmi_prior_np = _build_lsmi_profile_prior_from_raw(z_raw, tau=tau_lsmi_prior)["q"]

    model_idx = torch.tensor(data["model_id"].astype(str).map(model_to_idx).values, dtype=torch.long)
    item_idx = torch.tensor(data["item_id"].astype(str).map(item_to_idx).values, dtype=torch.long)
    s_image = torch.tensor(data["s_image"].astype(float).values, dtype=torch.float32)
    s_text = torch.tensor(data["s_text"].astype(float).values, dtype=torch.float32)
    y = torch.tensor(data["correct"].astype(float).values, dtype=torch.float32)

    train_idx, val_idx, num_val_items = _split_indices_grouped_by_item_model(data=data, val_ratio=val_ratio, seed=seed)
    train_ds = TensorDataset(model_idx[train_idx], item_idx[train_idx], s_image[train_idx], s_text[train_idx], y[train_idx])
    val_ds = TensorDataset(model_idx[val_idx], item_idx[val_idx], s_image[val_idx], s_text[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(1, len(train_ds))), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, len(val_ds))), shuffle=False) if len(val_ds) else None

    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"
    torch_device = torch.device(device)
    model = SRCoDModel(num_models=len(model_ids), z_item_std=z_item_std).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    ce_loss_fn = nn.BCEWithLogitsLoss()
    q_lsmi_prior = torch.tensor(q_lsmi_prior_np, dtype=torch.float32, device=torch_device)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_ce: list[float] = []
        epoch_kl: list[float] = []
        for mb_model, mb_item, mb_si, mb_st, mb_y in train_loader:
            mb_model = mb_model.to(torch_device)
            mb_item = mb_item.to(torch_device)
            mb_si = mb_si.to(torch_device)
            mb_st = mb_st.to(torch_device)
            mb_y = mb_y.to(torch_device)

            logits = model.logits(mb_model, mb_item, mb_si, mb_st)
            ce_loss = ce_loss_fn(logits, mb_y)
            profile = model.profile_from_mean_theta()
            q_model = profile["q"]
            profile_kl = torch.sum(
                q_lsmi_prior * (torch.log(q_lsmi_prior + 1e-12) - torch.log(q_model + 1e-12)),
                dim=1,
            ).mean()
            loss = ce_loss + float(lambda_profile_kl) * profile_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clamp_parameters()

            epoch_losses.append(float(loss.detach().cpu().item()))
            epoch_ce.append(float(ce_loss.detach().cpu().item()))
            epoch_kl.append(float(profile_kl.detach().cpu().item()))

        val_ce = _evaluate_val_bce(model, val_loader, ce_loss_fn, torch_device) if val_loader else float("nan")
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)),
            "train_ce": float(np.mean(epoch_ce)),
            "train_profile_kl": float(np.mean(epoch_kl)),
            "train_profile_kl_weighted": float(lambda_profile_kl * np.mean(epoch_kl)),
            "val_ce": val_ce,
        }
        history.append(record)
        if epoch == 1 or epoch == epochs or epoch % 25 == 0:
            LOGGER.info(
                "SRCoD epoch=%s train_loss=%.6f ce=%.6f kl=%.6f val_ce=%.6f",
                epoch,
                record["train_loss"],
                record["train_ce"],
                record["train_profile_kl_weighted"],
                val_ce,
            )

    item_scores = _export_item_scores(model, item_ids, q_lsmi_prior_np)
    model_scores = _export_model_scores(model, model_ids)
    summary = {
        "method": "SRCoD",
        "loss": "BCEWithLogitsLoss + lambda_profile_kl * KL(LSMI_prior || SRCoD_profile)",
        "num_models": len(model_ids),
        "item_count": len(item_ids),
        "num_response_rows": int(len(data)),
        "num_train_rows": int(len(train_idx)),
        "num_val_rows": int(len(val_idx)),
        "num_val_items": int(num_val_items),
        "seed": int(seed),
        "epochs": int(epochs),
        "lr": float(lr),
        "lambda_profile_kl": float(lambda_profile_kl),
        "tau_lsmi_prior": float(tau_lsmi_prior),
        "z_clip_quantile": float(z_clip_quantile),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_ce": history[-1]["val_ce"] if history else None,
    }
    return SRCoDTrainResult(history=history, item_scores=item_scores, model_scores=model_scores, summary=summary)


class SRCoDModel(nn.Module):
    def __init__(
        self,
        num_models: int,
        z_item_std: np.ndarray,
        theta_max: float = 4.0,
        theta_base_max: float = 0.1,
        difficulty_base_max: float = 4.0,
        difficulty_other_max: float = 4.0,
        a_value_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_models = int(num_models)
        self.k_dim = 4
        self.theta_max = float(theta_max)
        self.theta_base_max = float(theta_base_max)
        self.difficulty_base_max = float(difficulty_base_max)
        self.difficulty_other_max = float(difficulty_other_max)
        self.a_value_max = float(a_value_max)
        item_count = int(z_item_std.shape[0])

        self.raw_theta = nn.Parameter(torch.empty(num_models, self.k_dim))
        self.mix_m = nn.Parameter(torch.empty(self.k_dim, self.k_dim))
        self.mix_bias = nn.Parameter(torch.empty(self.k_dim))
        self.a_scale = nn.Parameter(torch.empty(self.k_dim))
        self.a_bias = nn.Parameter(torch.empty(self.k_dim))
        self.b_scale = nn.Parameter(torch.empty(self.k_dim))
        self.b_bias = nn.Parameter(torch.empty(self.k_dim))
        self.delta_a = nn.Parameter(torch.empty(item_count, self.k_dim))
        self.eps_b = nn.Parameter(torch.empty(item_count, self.k_dim))
        self.register_buffer("z_item_std", torch.tensor(z_item_std, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.raw_theta, mean=0.5, std=0.1)
        with torch.no_grad():
            self.raw_theta[:, 0].fill_(0.05)
        nn.init.normal_(self.mix_m, mean=0.0, std=0.02)
        with torch.no_grad():
            self.mix_m += torch.eye(self.k_dim)
        nn.init.normal_(self.mix_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.a_scale, mean=1.0, std=0.05)
        nn.init.normal_(self.a_bias, mean=0.0, std=0.05)
        nn.init.normal_(self.b_scale, mean=1.0, std=0.05)
        nn.init.normal_(self.b_bias, mean=0.0, std=0.05)
        nn.init.normal_(self.delta_a, mean=0.0, std=0.05)
        nn.init.normal_(self.eps_b, mean=0.0, std=0.05)
        self.clamp_parameters()

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            self.raw_theta[:, 0].clamp_(1e-4, self.theta_base_max)
            self.raw_theta[:, 1:].clamp_(1e-4, self.theta_max)

    def _project_h(self, item_idx: torch.Tensor | None = None) -> torch.Tensor:
        z = self.z_item_std if item_idx is None else self.z_item_std[item_idx]
        return z @ self.mix_m.T + self.mix_bias

    def item_parameters(self, item_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self._project_h(item_idx)
        delta_a = self.delta_a if item_idx is None else self.delta_a[item_idx]
        eps_b = self.eps_b if item_idx is None else self.eps_b[item_idx]
        raw_a = h * self.a_scale + self.a_bias + delta_a
        a = torch.clamp(torch.nn.functional.softplus(raw_a) + 1e-6, 1e-4, self.a_value_max)
        raw_b = h * self.b_scale + self.b_bias + eps_b
        b_base = torch.clamp(torch.nn.functional.softplus(raw_b[:, :1]) + 1e-6, 0.0, self.difficulty_base_max)
        b_other = torch.clamp(
            -(torch.nn.functional.softplus(raw_b[:, 1:]) + 1e-6),
            -self.difficulty_other_max,
            0.0,
        )
        return {"a": a, "b": torch.cat([b_base, b_other], dim=1)}

    def logits(
        self,
        model_idx: torch.Tensor,
        item_idx: torch.Tensor,
        s_image: torch.Tensor,
        s_text: torch.Tensor,
    ) -> torch.Tensor:
        theta = self.raw_theta[model_idx]
        params = self.item_parameters(item_idx)
        core = params["a"] * theta - params["b"]
        x = torch.stack([torch.ones_like(s_image), s_image, s_text, s_image * s_text], dim=1)
        return torch.sum(x * core, dim=1)

    def profile_from_mean_theta(self) -> dict[str, torch.Tensor]:
        params = self.item_parameters(None)
        theta_ref = self.raw_theta.mean(dim=0)
        return _profile_from_item_params_torch(params["a"], params["b"], theta_ref)


def _dim_logits(a: torch.Tensor, b: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return a * theta - b


def _condition_logits_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = _dim_logits(a[..., 0], b[..., 0], theta[..., 0])
    img = _dim_logits(a[..., 1], b[..., 1], theta[..., 1])
    txt = _dim_logits(a[..., 2], b[..., 2], theta[..., 2])
    cross = _dim_logits(a[..., 3], b[..., 3], theta[..., 3])
    return base, base + img, base + txt, base + img + txt + cross


def _profile_from_item_params_torch(a: torch.Tensor, b: torch.Tensor, theta_ref: torch.Tensor) -> dict[str, torch.Tensor]:
    theta = theta_ref.view(1, 4).expand(a.shape[0], 4)
    logit_none, logit_img, logit_txt, logit_both = _condition_logits_torch(a, b, theta)
    p_none = torch.sigmoid(logit_none)
    p_img = torch.sigmoid(logit_img)
    p_txt = torch.sigmoid(logit_txt)
    p_both = torch.sigmoid(logit_both)
    score_red = torch.minimum(p_img, p_txt) - p_none
    score_img = p_img - torch.maximum(p_txt, p_none)
    score_txt = p_txt - torch.maximum(p_img, p_none)
    score_syn = p_both - torch.maximum(p_img, p_txt)
    score = torch.stack([score_red, score_img, score_txt, score_syn], dim=1)
    q = torch.softmax(score, dim=1)
    return {
        "p_none": p_none,
        "p_img": p_img,
        "p_txt": p_txt,
        "p_both": p_both,
        "score_red": score_red,
        "score_img": score_img,
        "score_txt": score_txt,
        "score_syn": score_syn,
        "q": q,
    }


def _prepare_lsmi_item_features(lsmi_metrics: pd.DataFrame) -> pd.DataFrame:
    cols = LSMI_FEATURES
    missing = [c for c in ["question_id"] + cols if c not in lsmi_metrics.columns]
    if missing:
        raise ValueError(f"lsmi_metrics missing required columns: {missing}")
    out = lsmi_metrics[["question_id"] + cols].copy().drop_duplicates(subset=["question_id"], keep="last")
    out = out.rename(columns={"question_id": "item_id"})
    out["item_id"] = out["item_id"].astype(str)
    for old, new in zip(cols, [f"z_{name}" for name in Z_NAMES]):
        out = out.rename(columns={old: new})
    return out


def _clip_features(z_raw: np.ndarray, z_clip_quantile: float) -> np.ndarray:
    if not (0.5 < float(z_clip_quantile) < 1.0):
        return z_raw
    clipped = z_raw.copy()
    lo_q = 1.0 - float(z_clip_quantile)
    hi_q = float(z_clip_quantile)
    for dim in range(clipped.shape[1]):
        lo = np.quantile(clipped[:, dim], lo_q)
        hi = np.quantile(clipped[:, dim], hi_q)
        clipped[:, dim] = np.clip(clipped[:, dim], lo, hi)
    return clipped


def _row_softmax_np(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    ex = np.exp(arr)
    return ex / np.maximum(np.sum(ex, axis=1, keepdims=True), 1e-12)


def _robust_minmax_1d(x: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _build_lsmi_profile_prior_from_raw(z_raw: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    tau = max(float(tau), 1e-6)
    r = _robust_minmax_1d(z_raw[:, 0])
    u_i = _robust_minmax_1d(z_raw[:, 1])
    u_t = _robust_minmax_1d(z_raw[:, 2])
    s = _robust_minmax_1d(z_raw[:, 3])
    a_img = r + u_i
    a_txt = r + u_t
    a_both = r + u_i + u_t + s
    score = np.stack(
        [
            np.minimum(a_img, a_txt),
            np.maximum(a_img - a_txt, 0.0),
            np.maximum(a_txt - a_img, 0.0),
            np.maximum(a_both - np.maximum(a_img, a_txt), 0.0),
        ],
        axis=1,
    )
    return {"q": _row_softmax_np(score / tau)}


def _split_indices_grouped_by_item_model(data: pd.DataFrame, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    n_obs = len(data)
    if val_ratio <= 0.0 or n_obs < 10:
        return torch.arange(n_obs, dtype=torch.long), torch.empty(0, dtype=torch.long), 0
    key_df = data[["model_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.RandomState(seed)
    val_key_set: set[tuple[str, str]] = set()
    val_item_set: set[str] = set()
    for item_id, group in key_df.groupby("item_id", sort=False):
        candidates = group["model_id"].astype(str).tolist()
        if len(candidates) <= 1:
            continue
        desired = max(1, int(np.ceil(float(val_ratio) * len(candidates))))
        desired = min(desired, len(candidates) - 1)
        selected = rng.choice(np.array(candidates, dtype=object), size=desired, replace=False)
        for model_id in selected.tolist():
            val_key_set.add((str(model_id), str(item_id)))
            val_item_set.add(str(item_id))
    if not val_key_set:
        return torch.arange(n_obs, dtype=torch.long), torch.empty(0, dtype=torch.long), 0
    row_keys = list(zip(data["model_id"].astype(str), data["item_id"].astype(str)))
    val_mask = np.array([key in val_key_set for key in row_keys], dtype=bool)
    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask)
    if len(val_idx) == 0 or len(train_idx) == 0:
        return torch.arange(n_obs, dtype=torch.long), torch.empty(0, dtype=torch.long), 0
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long), len(val_item_set)


def _evaluate_val_bce(model: SRCoDModel, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for mb_model, mb_item, mb_si, mb_st, mb_y in val_loader:
            logits = model.logits(
                mb_model.to(device),
                mb_item.to(device),
                mb_si.to(device),
                mb_st.to(device),
            )
            losses.append(float(loss_fn(logits, mb_y.to(device)).detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def _export_item_scores(model: SRCoDModel, item_ids: list[str], q_lsmi_prior: np.ndarray) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        profile = model.profile_from_mean_theta()
        q = profile["q"].detach().cpu().numpy()
        out = pd.DataFrame(
            {
                "item_id": item_ids,
                "q_redundant": q[:, 0],
                "q_image_unique": q[:, 1],
                "q_text_unique": q[:, 2],
                "q_synergistic": q[:, 3],
                "pred_label": [LABELS[int(i)] for i in np.argmax(q, axis=1)],
                "lsmi_q_redundant": q_lsmi_prior[:, 0],
                "lsmi_q_image_unique": q_lsmi_prior[:, 1],
                "lsmi_q_text_unique": q_lsmi_prior[:, 2],
                "lsmi_q_synergistic": q_lsmi_prior[:, 3],
                "p_none": profile["p_none"].detach().cpu().numpy(),
                "p_image": profile["p_img"].detach().cpu().numpy(),
                "p_text": profile["p_txt"].detach().cpu().numpy(),
                "p_both": profile["p_both"].detach().cpu().numpy(),
            }
        )
    return out


def _export_model_scores(model: SRCoDModel, model_ids: list[str]) -> pd.DataFrame:
    with torch.no_grad():
        theta = model.raw_theta.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "model_id": model_ids,
            "theta_base": theta[:, 0],
            "theta_image": theta[:, 1],
            "theta_text": theta[:, 2],
            "theta_synergy": theta[:, 3],
        }
    )


def evaluate_item_labels(item_scores: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    if list(labels.columns) != ["item_id", "source_type_label"]:
        raise ValueError("labels must contain exactly item_id,source_type_label")
    label_id_col = "item_id"
    merged = item_scores[["item_id", "pred_label"]].merge(
        labels[[label_id_col, "source_type_label"]].rename(columns={label_id_col: "item_id"}),
        on="item_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No item_scores rows matched labels.")
    y_true = merged["source_type_label"].astype(str).tolist()
    y_pred = merged["pred_label"].astype(str).tolist()
    return {
        "num_eval_items": int(len(merged)),
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred, labels=LABELS),
    }


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum((t == label and p == label) for t, p in zip(y_true, y_pred))
        fp = sum((t != label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum((t == label and p != label) for t, p in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
