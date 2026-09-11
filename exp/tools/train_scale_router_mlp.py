import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_p2seg_with_sam2 import ROUTER_FEATURE_KEYS, ROUTER_FEATURE_KEYS_BY_VERSION


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def inverse_sqrt_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights *= float(num_classes) / float(weights.sum())
    return counts, weights


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def split_by_image(rows, val_ratio, seed):
    image_ids = sorted({str(row.get("image_id", "")) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(round(len(image_ids) * float(val_ratio)))) if len(image_ids) > 1 else 0
    val_ids = set(image_ids[:val_count])
    train_idx, val_idx = [], []
    for idx, row in enumerate(rows):
        (val_idx if str(row.get("image_id", "")) in val_ids else train_idx).append(idx)
    if not train_idx and val_idx:
        train_idx.append(val_idx.pop())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def weighted_losses(scale_p, route_p, fusion_pred, scale_t, route_t, fusion_y, sample_weight):
    row_weight = sample_weight[:, None] / float(max(sample_weight.sum(), 1e-6))
    scale_loss = -np.sum(row_weight * scale_t * np.log(np.maximum(scale_p, 1e-12)))
    route_loss = -np.sum(row_weight * route_t * np.log(np.maximum(route_p, 1e-12)))
    fusion_loss = np.sum(row_weight * 0.5 * (fusion_pred - fusion_y) ** 2)
    return float(scale_loss + route_loss + fusion_loss), float(scale_loss), float(route_loss), float(fusion_loss)


def accuracy(pred, target):
    if len(target) == 0:
        return 0.0
    return float(np.mean(pred == target))


def mae(pred, target):
    if len(target) == 0:
        return 0.0
    return float(np.mean(np.abs(pred.reshape(-1) - target.reshape(-1))))


def main():
    parser = argparse.ArgumentParser(description="Train ScaleAwareRouter MLP from pseudo labels.")
    parser.add_argument("jsonl", help="JSONL exported by export_scale_router_mlp_data.py.")
    parser.add_argument("out", help="Output router MLP JSON.")
    parser.add_argument("--router-head-version", type=int, choices=sorted(ROUTER_FEATURE_KEYS_BY_VERSION.keys()), default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-class-balance", action="store_true", help="Disable inverse-sqrt class reweighting for scale and route labels.")
    args = parser.parse_args()

    rows = load_jsonl(args.jsonl)
    if not rows:
        raise ValueError("No training rows found.")
    feature_keys = ROUTER_FEATURE_KEYS_BY_VERSION[int(args.router_head_version)]
    for row in rows:
        row_version = int(row.get("router_head_version", 1 if row.get("feature_keys") == ROUTER_FEATURE_KEYS else args.router_head_version))
        if row_version != int(args.router_head_version):
            raise ValueError(f"Router head version mismatch: row={row_version}, requested={args.router_head_version}")
        if row.get("feature_keys") != feature_keys:
            raise ValueError("Feature key order mismatch. Regenerate JSONL with the current exporter.")

    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    scale_y = np.asarray([int(row["scale_label"]) for row in rows], dtype=np.int64)
    route_y = np.asarray([int(row["route_label"]) for row in rows], dtype=np.int64)
    fusion_y = np.asarray([float(row.get("fusion_weight", 0.0)) for row in rows], dtype=np.float32)[:, None]
    release_y = np.asarray([float(row.get("release_score", 1.0 if int(row["route_label"]) != 0 else 0.0)) for row in rows], dtype=np.float32)[:, None]
    fusion_delta_y = np.asarray([float(row.get("fusion_weight_delta", max(0.0, float(row.get("fusion_weight", 0.0)) - 0.5))) for row in rows], dtype=np.float32)[:, None]
    train_idx, val_idx = split_by_image(rows, args.val_ratio, args.seed)

    if args.no_class_balance:
        sample_weight = np.ones(len(rows), dtype=np.float32)
        scale_counts = np.bincount(scale_y, minlength=3).astype(np.float32)
        route_counts = np.bincount(route_y, minlength=4).astype(np.float32)
        scale_class_weight = np.ones(3, dtype=np.float32)
        route_class_weight = np.ones(4, dtype=np.float32)
    else:
        scale_counts, scale_class_weight = inverse_sqrt_class_weights(scale_y, 3)
        route_counts, route_class_weight = inverse_sqrt_class_weights(route_y, 4)
        sample_weight = scale_class_weight[scale_y] * route_class_weight[route_y]
        sample_weight *= float(len(sample_weight)) / float(max(sample_weight.sum(), 1e-6))

    mean = x[train_idx].mean(axis=0)
    std = np.maximum(x[train_idx].std(axis=0), 1e-6)
    x_norm = (x - mean) / std
    n, in_dim = x.shape
    hidden_dim = int(args.hidden_dim)
    rng = np.random.default_rng(args.seed)

    hidden_w = rng.normal(0.0, 0.02, size=(hidden_dim, in_dim)).astype(np.float32)
    hidden_b = np.zeros(hidden_dim, dtype=np.float32)
    scale_w = rng.normal(0.0, 0.02, size=(3, hidden_dim)).astype(np.float32)
    scale_b = np.zeros(3, dtype=np.float32)
    route_w = rng.normal(0.0, 0.02, size=(4, hidden_dim)).astype(np.float32)
    route_b = np.zeros(4, dtype=np.float32)
    fusion_w = rng.normal(0.0, 0.02, size=(hidden_dim,)).astype(np.float32)
    fusion_b = np.asarray(0.0, dtype=np.float32)

    scale_t = one_hot(scale_y, 3)
    route_t = one_hot(route_y, 4)
    lr = float(args.lr)
    wd = float(args.weight_decay)

    best = None
    stale = 0
    for epoch in range(int(args.epochs)):
        tx = x_norm[train_idx]
        z = tx.dot(hidden_w.T) + hidden_b
        h = np.maximum(z, 0.0)
        scale_logits = h.dot(scale_w.T) + scale_b
        scale_p = softmax(scale_logits)
        row_weight = sample_weight[train_idx, None] / float(max(sample_weight[train_idx].sum(), 1e-6))
        d_scale = row_weight * (scale_p - scale_t[train_idx])

        if int(args.router_head_version) == 1:
            route_logits = h.dot(route_w.T) + route_b
            fusion_pred = sigmoid(h.dot(fusion_w)[:, None] + fusion_b)
            route_p = softmax(route_logits)
            d_route = row_weight * (route_p - route_t[train_idx])
            d_fusion = row_weight * (fusion_pred - fusion_y[train_idx]) * fusion_pred * (1.0 - fusion_pred)
            grad_route_w = d_route.T.dot(h) + wd * route_w
            grad_route_b = d_route.sum(axis=0)
            grad_fusion_w = d_fusion[:, 0].dot(h) + wd * fusion_w
            grad_fusion_b = d_fusion.sum()
            dh = d_scale.dot(scale_w) + d_route.dot(route_w) + d_fusion.dot(fusion_w[None, :])
        else:
            release_pred = sigmoid(h.dot(route_w[0])[:, None] + route_b[0])
            fusion_delta_pred = np.tanh(h.dot(fusion_w)[:, None] + fusion_b)
            d_release = row_weight * (release_pred - release_y[train_idx]) * release_pred * (1.0 - release_pred)
            d_delta = row_weight * (fusion_delta_pred - fusion_delta_y[train_idx]) * (1.0 - fusion_delta_pred ** 2)
            grad_route_w = np.zeros_like(route_w)
            grad_route_b = np.zeros_like(route_b)
            grad_route_w[0] = d_release[:, 0].dot(h) + wd * route_w[0]
            grad_route_b[0] = d_release.sum()
            grad_fusion_w = d_delta[:, 0].dot(h) + wd * fusion_w
            grad_fusion_b = d_delta.sum()
            dh = d_scale.dot(scale_w) + d_release.dot(route_w[0][None, :]) + d_delta.dot(fusion_w[None, :])

        grad_scale_w = d_scale.T.dot(h) + wd * scale_w
        grad_scale_b = d_scale.sum(axis=0)
        dz = dh * (z > 0)
        grad_hidden_w = dz.T.dot(tx) + wd * hidden_w
        grad_hidden_b = dz.sum(axis=0)

        hidden_w -= lr * grad_hidden_w
        hidden_b -= lr * grad_hidden_b
        scale_w -= lr * grad_scale_w
        scale_b -= lr * grad_scale_b
        route_w -= lr * grad_route_w
        route_b -= lr * grad_route_b
        fusion_w -= lr * grad_fusion_w
        fusion_b -= lr * grad_fusion_b

        eval_idx = val_idx if len(val_idx) else train_idx
        ez = x_norm[eval_idx].dot(hidden_w.T) + hidden_b
        eh = np.maximum(ez, 0.0)
        escale = softmax(eh.dot(scale_w.T) + scale_b)
        if int(args.router_head_version) == 1:
            eroute = softmax(eh.dot(route_w.T) + route_b)
            efusion = sigmoid(eh.dot(fusion_w)[:, None] + fusion_b)
            val_loss = weighted_losses(escale, eroute, efusion, scale_t[eval_idx], route_t[eval_idx], fusion_y[eval_idx], sample_weight[eval_idx])[0]
            metrics = {
                "scale_acc": accuracy(escale.argmax(axis=1), scale_y[eval_idx]),
                "route_acc": accuracy(eroute.argmax(axis=1), route_y[eval_idx]),
                "fusion_mae": mae(efusion, fusion_y[eval_idx]),
            }
        else:
            erelease = sigmoid(eh.dot(route_w[0])[:, None] + route_b[0])
            edelta = np.tanh(eh.dot(fusion_w)[:, None] + fusion_b)
            row_weight_eval = sample_weight[eval_idx, None] / float(max(sample_weight[eval_idx].sum(), 1e-6))
            val_loss = float(
                -np.sum(row_weight_eval * scale_t[eval_idx] * np.log(np.maximum(escale, 1e-12)))
                + np.sum(row_weight_eval * 0.5 * (erelease - release_y[eval_idx]) ** 2)
                + np.sum(row_weight_eval * 0.5 * (edelta - fusion_delta_y[eval_idx]) ** 2)
            )
            metrics = {
                "scale_acc": accuracy(escale.argmax(axis=1), scale_y[eval_idx]),
                "release_mae": mae(erelease, release_y[eval_idx]),
                "fusion_delta_mae": mae(edelta, fusion_delta_y[eval_idx]),
            }
        if best is None or val_loss < best["val_loss"]:
            best = {
                "epoch": int(epoch),
                "val_loss": float(val_loss),
                "metrics": metrics,
                "hidden_w": hidden_w.copy(),
                "hidden_b": hidden_b.copy(),
                "scale_w": scale_w.copy(),
                "scale_b": scale_b.copy(),
                "route_w": route_w.copy(),
                "route_b": route_b.copy(),
                "fusion_w": fusion_w.copy(),
                "fusion_b": np.asarray(fusion_b).copy(),
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    hidden_w = best["hidden_w"]
    hidden_b = best["hidden_b"]
    scale_w = best["scale_w"]
    scale_b = best["scale_b"]
    route_w = best["route_w"]
    route_b = best["route_b"]
    fusion_w = best["fusion_w"]
    fusion_b = best["fusion_b"]

    model = {
        "router_head_version": int(args.router_head_version),
        "feature_keys": feature_keys,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "hidden_weight": hidden_w.tolist(),
        "hidden_bias": hidden_b.tolist(),
        "scale_weight": scale_w.tolist(),
        "scale_bias": scale_b.tolist(),
        "num_rows": int(n),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "best_epoch": int(best["epoch"]),
        "validation": {
            "loss": float(best["val_loss"]),
            **best["metrics"],
        },
        "class_balance": {
            "enabled": not bool(args.no_class_balance),
            "scale_counts": scale_counts.tolist(),
            "route_counts": route_counts.tolist(),
            "scale_class_weight": scale_class_weight.tolist(),
            "route_class_weight": route_class_weight.tolist(),
        },
    }
    if int(args.router_head_version) == 1:
        model.update({
            "route_weight": route_w.tolist(),
            "route_bias": route_b.tolist(),
            "fusion_weight": fusion_w.tolist(),
            "fusion_bias": float(fusion_b),
        })
    else:
        model.update({
            "release_weight": route_w[0].tolist(),
            "release_bias": float(route_b[0]),
            "fusion_delta_weight": fusion_w.tolist(),
            "fusion_delta_bias": float(fusion_b),
        })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(model, f)
    print(
        "trained_rows={n}, train_rows={train}, val_rows={val}, hidden_dim={hidden}, "
        "version={version}, best_epoch={epoch}, val_loss={loss:.6f}, metrics={metrics}, output={out}".format(
            n=n,
            train=len(train_idx),
            val=len(val_idx),
            hidden=hidden_dim,
            version=int(args.router_head_version),
            epoch=best["epoch"],
            loss=best["val_loss"],
            metrics=json.dumps(best["metrics"], sort_keys=True),
            out=args.out,
        )
    )


if __name__ == "__main__":
    main()
