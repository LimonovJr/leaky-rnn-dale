# Leaky RNN with Dale's Law — V3

Biologically-constrained leaky RNN trained on a cued visuospatial detection task with distractors.
Implements E/I split, Dale's law, sparse masks, and recurrent noise scaling.

---

## Model

```
h_{t+1} = (1 - α) h_t + α · φ(W_rec_eff h_t + W_in_eff x_t + b_h + ξ_t)
α = dt / τ,   ξ_t ~ N(0, σ_eff²),   σ_eff = √(2/α) · σ_rec
```

Dale's law: `W_rec` stores unsigned magnitudes; a fixed `ei_sign` vector (+1/−1) enforces sign constraints per sender column throughout training.

---

## Task: `CuedTargetWithDistractorsV3`

Observation space `(7,)`:

| Channel | Meaning |
|---------|---------|
| 0 | fixation |
| 1 | cue_on |
| 2–3 | cue (x, y) — near-central (±0.4), scaled by `cue_strength` |
| 4 | stim_on |
| 5–6 | stimulus (x, y) — peripheral (±1.0), scaled by strength |

Trial structure: **fixation** → **cue** (350 ms) → **delay/CTOA** (Beta(2.2,1.6) → [1000, 3300] ms) → **target** (100 ms) → **post-target** (900 ms).

---

## Training curriculum

| Stage | `cue_strength` | `p_distractor` | Updates | Early stopping |
|-------|---------------|----------------|---------|----------------|
| 0 | 0.0 | 0.0 | 1 000 | off |
| 1 | 1.0 | 0.0 | 1 000 | off |
| 2 | 1.0 | 0.6 | until early stop | on (default) |

Early stopping in stage 2: halts when `p_miss == 0` for 3 consecutive print-steps, restores weights from 5 steps back.

---

## Installation

```bash
conda create -n rnn-env python=3.10 -y
conda activate rnn-env
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install neurogym @ git+https://github.com/neurogym/neurogym.git
pip install numpy matplotlib scikit-learn
```

> **Note:** for `hidden_size=128` / `batch_size=64`, CPU is faster than GPU — training runs on `device = 'cpu'` by default.

---

## Repository layout

```
leaky-rnn-dale/
├── src/
│   ├── env.py        — CuedTargetWithDistractorsV3
│   ├── model.py      — BioLeakyRNN
│   ├── training.py   — TrainConfig, train_supervised (with early stopping)
│   ├── dataset.py    — rollout, batch factory
│   ├── analysis.py   — collect_trials, PCA, dPCA, spatial separation
│   └── plotting.py   — visualisation helpers
├── notebooks/
│   ├── 01_train.ipynb   — 3-stage curriculum training
│   └── 02_analysis.ipynb — PCA, dPCA, spatial analysis
├── checkpoints/      (.gitignored)
└── requirements.txt
```
