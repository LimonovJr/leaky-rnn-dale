"""
Supervised training for the (x, y) continuous-output spatial task.

The model's primary output head (W_out, shape [2, hidden]) is re-interpreted
as a linear (x, y) regression head — no softmax, no argmax. Loss is masked
MSE between model output and target_pos at every post-cue timestep, with a
short settle period to let the leaky dynamics integrate the cue.

Separate from train_supervised (binary task) so the two paths don't share
config or print formatting.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from src.dataset import make_train_batch
from src.training import masked_spatial_mse, masked_temporal_cross_entropy


@dataclass
class SpatialTrainConfig:
    batch_size:    int   = 64
    lr:            float = 1e-3
    max_updates:   int   = 5000
    print_every:   int   = 50
    grad_clip:     float = 10.0
    # First settle_steps post-cue timesteps excluded from the loss — the env
    # has a fixation epoch and a brief cue epoch, but the post-cue dynamics
    # still need ~5 dt=20ms steps (100ms, ~67% settled at alpha=0.2) to start
    # tracking target_pos meaningfully.
    settle_steps:  int   = 5
    # Multi-task weights: model output is [B,T,4] = (x,y) position head (W_out[:2],
    # MSE) + hold/release logits head (W_out[2:], CE). The CE head re-imposes the
    # response-timing demand the pure-spatial task lacked — restoring the "when"
    # (temporal) computation.
    pos_weight:    float = 1.0
    ce_weight:     float = 1.0
    device:        str   = "cpu"
    # Quick-eval bookkeeping: each print step, roll out `eval_n_trials` fresh
    # trials to compute hit / miss / false_alarm rates (same outcome logic as
    # collect_trials_spatial in src/analysis.py). Cost: a single inference pass
    # on `eval_n_trials` trials per print step (~1s for 32 trials at B=180,
    # T~250). Set to 0 to disable eval and print only the training MSE.
    eval_n_trials: int   = 32


def train_supervised_spatial(model, env_fn, cfg: SpatialTrainConfig):
    """MSE supervision on the model's PRIMARY output (W_out as (x, y) head)."""
    if not getattr(model, "batch_first", True):
        raise ValueError(
            "train_supervised_spatial expects model.batch_first=True; "
            f"got {model.batch_first}"
        )

    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(0.9, 0.999))

    history = {k: [] for k in (
        "loss", "update",
        "p_hit", "p_miss", "p_fa",
        "mean_err", "rt_ms",
    )}
    checked_shapes = False
    # Imported lazily to avoid circular dependency with src.analysis,
    # which itself doesn't depend on this file but the import would
    # add overhead for callers that just want SpatialTrainConfig.
    from src.analysis import collect_trials_spatial

    for upd in range(1, cfg.max_updates + 1):
        model.train()

        x, y, mask, _fa, xy_true, xy_mask, _lengths = make_train_batch(
            env_fn=env_fn,
            batch_size=cfg.batch_size,
            dt=int(model.dt),
            device=cfg.device,
        )

        # Multi-task head: out[..., :2] = (x, y) position; out[..., 2:4] =
        # hold/release logits. No softmax/argmax here (CE applies it).
        out, _, _h_seq = model(x, return_hidden=True)
        xy_pred = out[..., :2]
        logits  = out[..., 2:4]

        if not checked_shapes:
            B, T, _ = x.shape
            assert out.shape == (B, T, 4), \
                f"model output shape {tuple(out.shape)} != (B, T, 4) — set output_size=4"
            assert xy_true.shape == (B, T, 2), \
                f"xy_true shape {tuple(xy_true.shape)} != (B, T, 2)"
            assert xy_mask.shape == (B, T), \
                f"xy_mask shape {tuple(xy_mask.shape)} != (B, T)"
            checked_shapes = True

        if cfg.settle_steps > 0:
            settle = torch.ones_like(xy_mask)
            settle[:, : cfg.settle_steps] = 0.0
            loss_mask = xy_mask * settle
        else:
            loss_mask = xy_mask

        loss_pos = masked_spatial_mse(xy_pred, xy_true, loss_mask)
        loss_ce  = masked_temporal_cross_entropy(logits, y, mask)
        loss = cfg.pos_weight * loss_pos + cfg.ce_weight * loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if upd % cfg.print_every == 0 or upd == 1:
            history["loss"].append(loss.detach().item())
            history["update"].append(upd)

            if cfg.eval_n_trials > 0:
                # Quick eval — fresh trials, model.eval() inside collect_*.
                eval_trials = collect_trials_spatial(
                    model, env_fn, n_trials=cfg.eval_n_trials,
                    device=cfg.device,
                )
                out_counts = Counter(t["train_outcome"] for t in eval_trials)
                n = len(eval_trials)
                p_hit  = out_counts.get("correct",     0) / n
                p_miss = out_counts.get("miss",        0) / n
                p_fa   = out_counts.get("false_alarm", 0) / n

                errs = [t["mean_err"] for t in eval_trials
                        if t["mean_err"] is not None]
                mean_err_val = float(np.mean(errs)) if errs else float("nan")

                rts = [t["rt_ms"] for t in eval_trials
                       if t["rt_ms"] is not None]
                rt_val = float(np.mean(rts)) if rts else float("nan")

                history["p_hit"].append(p_hit)
                history["p_miss"].append(p_miss)
                history["p_fa"].append(p_fa)
                history["mean_err"].append(mean_err_val)
                history["rt_ms"].append(rt_val)

                rt_str = f"{rt_val:5.0f}ms" if not np.isnan(rt_val) else "  n/a"
                err_str = f"{mean_err_val:5.3f}" if not np.isnan(mean_err_val) else " n/a"
                print(
                    f"Upd {upd:5d}/{cfg.max_updates} | "
                    f"L {loss.detach().item():.4f} pos {loss_pos.detach().item():.4f} "
                    f"ce {loss_ce.detach().item():.4f} | err {err_str} | "
                    f"hit {p_hit*100:5.1f}%  miss {p_miss*100:5.1f}%  "
                    f"FA {p_fa*100:5.1f}% | RT {rt_str} (n={n})"
                )
                # Switch model back to train mode (collect_trials_spatial
                # called model.eval()).
                model.train()
            else:
                for k in ("p_hit", "p_miss", "p_fa", "mean_err", "rt_ms"):
                    history[k].append(float("nan"))
                print(f"Upd {upd:5d}/{cfg.max_updates} | "
                      f"L {loss.detach().item():.4f} pos {loss_pos.detach().item():.4f} "
                      f"ce {loss_ce.detach().item():.4f}")

    return history
