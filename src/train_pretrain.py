"""
Spatial pretraining: MSE on (x, y) decoded from h_seq via model.decode_xy,
plus the same masked_com_loss + sparsity_loss used by the main loop. The
COM term is what makes the pretrained representation topographic on the
sheet — without it, MSE alone admits any linear-decodable spatial code,
and stage 2's COM term would then erase the pretrained representation.

Replaces the stage-0/stage-1 curriculum with a single supervised
pretraining pass on SpatialPretrain (cue at random continuous (x, y),
on for the whole trial).
"""

from dataclasses import dataclass

import torch

from src.dataset import make_train_batch
from src.training import masked_spatial_mse, masked_com_loss, sparsity_loss


@dataclass
class PretrainConfig:
    batch_size:       int   = 64
    lr:               float = 1e-3
    max_updates:      int   = 2000
    print_every:      int   = 50
    grad_clip:        float = 10.0
    # First settle_steps timesteps of every trial are excluded from the
    # MSE/COM masks: with dt=20ms, tau=100ms => alpha=0.2 and h reaches
    # ~89% of steady state only after ~10 steps.
    settle_steps:     int   = 10
    # Loss weights — mse_weight=1.0 is the historical scale; com_weight
    # and sparsity_weight mirror src.training.TrainConfig defaults so the
    # regularization regime is identical to stage 2.
    mse_weight:       float = 1.0
    com_weight:       float = 0.5
    sparsity_weight:  float = 0.01
    device:           str   = "cpu"


def train_spatial_pretrain(model, env_fn, cfg: PretrainConfig):
    """Multi-term pretraining (MSE + COM + sparsity) that produces a
    topographic spatial code transferable to stage 2."""
    if not getattr(model, "batch_first", True):
        raise ValueError(
            "train_spatial_pretrain expects model.batch_first=True; "
            f"got {model.batch_first}"
        )
    if not hasattr(model, "coords"):
        raise ValueError(
            "model has no .coords buffer — COM loss requires a topographic "
            "model like BioLeakyRNNTopo."
        )

    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(0.9, 0.999))

    history = {k: [] for k in ("loss", "mse", "com", "sparsity", "update")}
    checked_shapes = False

    for upd in range(1, cfg.max_updates + 1):
        model.train()

        x, _y, _m, _fa, xy_true, xy_mask, _lengths = make_train_batch(
            env_fn=env_fn,
            batch_size=cfg.batch_size,
            dt=int(model.dt),
            device=cfg.device,
        )

        _, _, h_seq = model(x, return_hidden=True)
        xy_pred = model.decode_xy(h_seq)

        if not checked_shapes:
            B, T, _ = x.shape
            assert h_seq.shape == (B, T, model.hidden_size), \
                f"h_seq shape {tuple(h_seq.shape)} != (B, T, H)=({B}, {T}, {model.hidden_size})"
            assert xy_pred.shape == (B, T, 2), \
                f"xy_pred shape {tuple(xy_pred.shape)} != (B, T, 2)"
            assert xy_true.shape == (B, T, 2), \
                f"xy_true shape {tuple(xy_true.shape)} != (B, T, 2)"
            checked_shapes = True

        if cfg.settle_steps > 0:
            settle = torch.ones_like(xy_mask)
            settle[:, : cfg.settle_steps] = 0.0
            loss_mask = xy_mask * settle
        else:
            loss_mask = xy_mask

        loss_mse  = masked_spatial_mse(xy_pred, xy_true, loss_mask)
        loss_com  = masked_com_loss(h_seq, model.coords, xy_true, loss_mask,
                                    period=2.0)
        loss_spar = sparsity_loss(h_seq) if cfg.sparsity_weight > 0.0 \
                    else torch.zeros((), device=h_seq.device)

        loss = (cfg.mse_weight * loss_mse
                + cfg.com_weight * loss_com
                + cfg.sparsity_weight * loss_spar)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if upd % cfg.print_every == 0 or upd == 1:
            history["loss"].append(loss.detach().item())
            history["mse"].append(loss_mse.detach().item())
            history["com"].append(loss_com.detach().item())
            history["sparsity"].append(loss_spar.detach().item())
            history["update"].append(upd)
            print(f"[pretrain] upd {upd:5d}  loss={loss.item():.4f}  "
                  f"mse={loss_mse.item():.4f}  com={loss_com.item():.4f}  "
                  f"sp={loss_spar.item():.4f}")

    return history
