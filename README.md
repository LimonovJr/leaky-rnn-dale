# Leaky RNN with Dale's Law

Biologically-constrained leaky recurrent neural network (RNN) trained on a cued
detection task with distractors. Implements E/I population split, Dale's law on
recurrent weights, sparse connectivity masks, and article-style recurrent noise
scaling.

---

## Model

`BioLeakyRNN` — discrete-time leaky RNN:

```
h_{t+1} = (1 - α) h_t + α · φ(W_rec_eff h_t + W_in_eff x_t + b_h + ξ_t)
y_t     = W_out_eff h_t + b_out

α = dt / τ
ξ_t ~ N(0, σ_eff²),  σ_eff = √(2/α) · σ_rec
```

**Dale's law** is enforced by storing unsigned magnitudes in `W_rec` and
multiplying each sender-column by a fixed ±1 sign vector (`ei_sign`).
This keeps the sign constraints hard throughout training.

---

## Task: `CuedTargetWithDistractorsV2`

A cued visuospatial detection task built on [NeuroGym](https://github.com/neurogym/neurogym).

| Period     | Duration                          |
|------------|-----------------------------------|
| Fixation   | jitter ~ Uniform(300, 600) ms     |
| Cue        | 350 ms                            |
| Delay      | CTOA ~ Beta(2.2, 1.6) → [1000, 3300] ms |
| Target     | 100 ms                            |
| Post-target| 900 ms                            |

Observation space: `(9,)` — [fixation, stim×4, cue×4]  
Action space: Discrete(2) — 0=hold, 1=release

Distractors appear during the delay period with a decaying hazard rate.

---

## Training stages

| Stage | `cue_strength` | `p_distractor_trial` | Updates |
|-------|---------------|----------------------|---------|
| 0     | 0.0           | 0.0                  | 1000    |
| 1     | 1.0           | 0.0                  | 1000    |
| 2     | 1.0           | 0.6                  | 3000    |

---

## Installation

```bash
git clone https://github.com/<your-username>/leaky-rnn-dale.git
cd leaky-rnn-dale
pip install -r requirements.txt
```

---

## Usage

### Notebooks

| Notebook | Content |
|----------|---------|
| `notebooks/01_train.ipynb` | Stage 0 → 1 → 2 training |
| `notebooks/02_analysis.ipynb` | PCA trajectories, dPCA, group comparisons |

### Quick start (Python)

```python
import torch
from src import BioLeakyRNN, CuedTargetWithDistractorsV2, TrainConfig, train_supervised

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BioLeakyRNN(
    input_size=9, hidden_size=128, output_size=2,
    dt=20.0, tau=100.0, activation="softplus",
    use_ei=True, exc_ratio=0.7, use_dale=True,
    mask_seed=42,
).to(device)

def make_env():
    return CuedTargetWithDistractorsV2(dt=20, cue_strength=0.0)

cfg = TrainConfig(batch_size=64, lr=1e-3, max_updates=1000, device=device)
history = train_supervised(model, make_env, cfg)

torch.save({"state_dict": model.state_dict()}, "checkpoints/stage0.pt")
```

---

## Repository structure

```
leaky-rnn-dale/
├── src/
│   ├── model.py      — BioLeakyRNN
│   ├── env.py        — CuedTargetWithDistractorsV2
│   ├── training.py   — loss functions, train loop, TrainConfig
│   ├── dataset.py    — rollout, batch factory
│   ├── analysis.py   — PCA, dPCA, trial selection
│   └── plotting.py   — all visualisation functions
├── notebooks/
│   ├── 01_train.ipynb
│   └── 02_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## References

- Mante, V., Sussillo, D., Shenoy, K. V., & Newsome, W. T. (2013).
  Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature*.
- Perez-Nieves, N., et al. (2021). Neural heterogeneity promotes robust learning. *Nature Communications*.
- NeuroGym: https://github.com/neurogym/neurogym
