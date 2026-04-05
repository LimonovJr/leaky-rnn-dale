# Leaky RNN with Dale's Law — V3

Biologically-constrained leaky RNN trained on a cued visuospatial detection task
with distractors. Implements E/I split, Dale's law, sparse masks, and
article-style recurrent noise scaling.

---

## Model

```
h_{t+1} = (1 - α) h_t + α · φ(W_rec_eff h_t + W_in_eff x_t + b_h + ξ_t)
y_t     = W_out_eff h_t + b_out

α       = dt / τ
ξ_t     ~ N(0, σ_eff²),   σ_eff = √(2/α) · σ_rec
```

Dale's law: `W_rec` stores unsigned magnitudes; a fixed `ei_sign` vector
(+1 / −1) multiplied per sender column enforces sign constraints throughout training.

---

## Task: `CuedTargetWithDistractorsV3`

| Obs channel | Meaning |
|-------------|---------|
| 0 | fixation |
| 1 | cue_on |
| 2–3 | cue (x, y) — near-central, scaled by `cue_strength` |
| 4 | stim_on |
| 5–6 | stimulus (x, y) — peripheral, scaled by `target/distractor_strength` |

**V3 vs V2:** continuous spatial coordinates replace one-hot location channels.
4 locations defined by quadrant (±1, ±1); cue uses inner coordinates (±0.4, ±0.4).

| Period | Duration |
|--------|----------|
| Fixation | jitter ~ U(700, 1200) ms |
| Cue | 350 ms |
| Delay (CTOA) | Beta(2.2, 1.6) → [1000, 3300] ms |
| Target | 100 ms |
| Post-target | 900 ms |

---

## Training stages

| Stage | `cue_strength` | `p_distractor` | Updates |
|-------|---------------|----------------|---------|
| 0 | 0.0 | 0.0 | 1 000 |
| 1 | 1.0 | 0.0 | 1 000 |
| 2 | 1.0 | 0.6 | 8 700 |

---

## Installation

```bash
git clone https://github.com/<your-username>/leaky-rnn-dale.git
cd leaky-rnn-dale
pip install -r requirements.txt
```

---

## Quick start

```python
import torch
from src import BioLeakyRNN, CuedTargetWithDistractorsV3, TrainConfig, train_supervised

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BioLeakyRNN(input_size=7, hidden_size=128, output_size=2,
                    dt=20.0, tau=100.0, use_ei=True, use_dale=True,
                    mask_seed=42).to(device)

def make_env(): return CuedTargetWithDistractorsV3(dt=20, cue_strength=0.0)

history = train_supervised(model, make_env,
                           TrainConfig(max_updates=1000, device=device))
torch.save({'state_dict': model.state_dict()}, 'checkpoints/stage0.pt')
```

---

## Repository layout

```
leaky-rnn-dale/
├── src/
│   ├── env.py        — CuedTargetWithDistractorsV3
│   ├── model.py      — BioLeakyRNN
│   ├── training.py   — losses, train loop, TrainConfig
│   ├── dataset.py    — rollout, batch factory
│   ├── analysis.py   — PCA, dPCA, spatial separation, trial collection
│   └── plotting.py   — visualisation
├── notebooks/
│   ├── 01_train.ipynb
│   └── 02_analysis.ipynb
├── checkpoints/      (.gitignored)
├── requirements.txt
└── README.md
```

---

## References

- Mante et al. (2013). Context-dependent computation by recurrent dynamics. *Nature*.
- Perez-Nieves et al. (2021). Neural heterogeneity promotes robust learning. *Nat. Commun.*
- NeuroGym: https://github.com/neurogym/neurogym
