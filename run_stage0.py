"""Run stage 0 training (1000 updates) and save training curve."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import BioLeakyRNN, CuedTargetWithDistractorsV2, TrainConfig, train_supervised

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

model = BioLeakyRNN(
    input_size=9, hidden_size=128, output_size=2,
    dt=20.0, tau=100.0, activation='softplus', sigma_rec=0.05,
    rec_init='diag', use_ei=True, exc_ratio=0.7, use_dale=True,
    rec_sparsity=0.0, in_sparsity=0.0, out_sparsity=0.0,
    allow_self_connections=True, mask_seed=42, dale_on_output=False,
).to(device)

def make_env_stage0():
    return CuedTargetWithDistractorsV2(dt=20, cue_strength=0.0,
                                       p_distractor_trial=0.0, distractor_strength=0.0)

cfg0 = TrainConfig(batch_size=64, lr=1e-3, max_updates=1000,
                   print_every=50, device=device)

history0 = train_supervised(model, make_env_stage0, cfg0)

torch.save({'state_dict': model.state_dict()}, 'checkpoints/stage0.pt')
print('\nCheckpoint saved: checkpoints/stage0.pt')

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='valid')

updates = np.arange(1, len(history0['loss']) + 1)
sw = 20  # smoothing window

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Stage 0 — simple detection (no cue, no distractors)', fontsize=13)

ax = axes[0]
ax.plot(updates, history0['loss'], alpha=0.25, color='steelblue', lw=0.8)
ax.plot(updates[sw-1:], smooth(history0['loss'], sw), color='steelblue', lw=2, label='total loss')
ax.plot(updates, history0['ce'], alpha=0.25, color='tomato', lw=0.8)
ax.plot(updates[sw-1:], smooth(history0['ce'], sw), color='tomato', lw=2, label='CE')
ax.set_xlabel('update')
ax.set_ylabel('loss')
ax.set_title('Loss')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
for key, color, label in [
    ('p_correct', 'green',  'correct'),
    ('p_miss',    'orange', 'miss'),
    ('p_abort',   'red',    'abort'),
]:
    vals = history0[key]
    ax.plot(updates, vals, alpha=0.2, color=color, lw=0.8)
    ax.plot(updates[sw-1:], smooth(vals, sw), color=color, lw=2, label=label)

ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('update')
ax.set_ylabel('fraction of trials')
ax.set_title('Trial outcomes')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = 'checkpoints/stage0_curve.png'
fig.savefig(out_path, dpi=130)
print(f'Curve saved: {out_path}')
