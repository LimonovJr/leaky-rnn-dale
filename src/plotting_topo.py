"""
Visualization helpers for BioLeakyRNNTopo.

The model no longer has learnable spatial input weights — spatial channels
(cue XYS, stim XYS) are consumed by a geometric Gaussian receptive field.
Instead of plotting W_in_topo, we visualize the RF drive evoked by a
synthetic stimulus, plus the E/I sheet layout.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_rf_drive(model, stim_xy=(1.0, 1.0), strength=1.0,
                  figsize=(5.0, 4.5), title=None, close_prior=False):
    """Per-neuron Gaussian RF drive for a test stimulus at stim_xy in [-1,+1]^2."""
    if close_prior:
        plt.close("all")

    assert hasattr(model, "coords"), "model must have a `coords` buffer (sheet)"
    coords = model.coords.detach().cpu().numpy()
    n_exc  = int(model.n_exc)
    sigma  = float(model.rf_sigma)

    x, y = float(stim_xy[0]), float(stim_xy[1])
    # Toroidal wrap (period 2.0 in [-1, +1]) — matches model._gaussian_rf_drive
    dx = coords[:, 0] - x; dx = dx - 2.0 * np.round(dx / 2.0)
    dy = coords[:, 1] - y; dy = dy - 2.0 * np.round(dy / 2.0)
    d2 = dx ** 2 + dy ** 2
    drive = strength * np.exp(-d2 / (2.0 * sigma * sigma))  # [H]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sc_e = ax.scatter(coords[:n_exc, 0], coords[:n_exc, 1],
                      c=drive[:n_exc], cmap="viridis",
                      vmin=0, vmax=max(drive.max(), 1e-6),
                      s=60, marker="o", edgecolor="k", linewidth=0.3, label="E")
    ax.scatter(coords[n_exc:, 0], coords[n_exc:, 1],
               c=drive[n_exc:], cmap="viridis",
               vmin=0, vmax=max(drive.max(), 1e-6),
               s=80, marker="^", edgecolor="k", linewidth=0.3, label="I")
    ax.plot([x], [y], "r*", markersize=14, label="stim")

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    if title is None:
        title = f"RF drive target ({x:.2f}, {y:.2f}), sigma={sigma}"
    ax.set_title(title)
    plt.colorbar(sc_e, ax=ax, fraction=0.046, label="drive")
    fig.tight_layout()
    return fig


def plot_sheet_layout(model, figsize=(5.0, 4.5), close_prior=False):
    """Scatter of neuron positions, colored by E/I identity."""
    if close_prior:
        plt.close("all")
    coords = model.coords.detach().cpu().numpy()
    n_exc = int(model.n_exc)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(coords[:n_exc, 0], coords[:n_exc, 1],
               s=45, marker="o", c="tab:blue", edgecolor="k", linewidth=0.3,
               label=f"E ({n_exc})")
    ax.scatter(coords[n_exc:, 0], coords[n_exc:, 1],
               s=70, marker="^", c="tab:red", edgecolor="k", linewidth=0.3,
               label=f"I ({coords.shape[0] - n_exc})")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Sheet layout ({coords.shape[0]} neurons)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_fix_weights(model, figsize=(5.0, 4.5), close_prior=False):
    """Scatter of W_in_fix (fixation channel weights) over the sheet."""
    if close_prior:
        plt.close("all")
    coords = model.coords.detach().cpu().numpy()
    w = model.W_in_fix.detach().cpu().numpy()
    vmax = float(np.abs(w).max()) or 1.0
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n_exc = int(model.n_exc)
    ax.scatter(coords[:n_exc, 0], coords[:n_exc, 1],
               c=w[:n_exc], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               s=60, marker="o", edgecolor="k", linewidth=0.3, label="E")
    sc = ax.scatter(coords[n_exc:, 0], coords[n_exc:, 1],
                    c=w[n_exc:], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=80, marker="^", edgecolor="k", linewidth=0.3, label="I")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("W_in_fix (fixation channel weights)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
