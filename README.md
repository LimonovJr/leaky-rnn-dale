# Leaky RNN with Dale's Law

Biologically-constrained leaky RNN trained on a cued visuospatial detection task.
Implements E/I split, Dale's law, sparse connectivity masks, and recurrent noise scaling.

## Model

```
h_{t+1} = (1 - α) h_t + α · φ(W_rec_eff h_t + W_in_eff x_t + b_h + ξ_t)
α = dt / τ,   ξ_t ~ N(0, σ_eff²),   σ_eff = √(2/α) · σ_rec
```

`W_rec` stores unsigned magnitudes; a fixed `ei_sign` vector (+1/−1) enforces per-sender
sign constraints throughout training.

## Task

`CuedTargetWithDistractorsV3` — cued visuospatial detection with distractors.

Input `(7,)`: fixation, cue_on, cue (x,y), stim_on, stimulus (x,y).

Trial: fixation → cue (350 ms) → delay (Beta(2.2,1.6) → [1000, 3300] ms)
       → target (100 ms) → post-target (900 ms).

## Training

| Stage | cue_strength | p_distractor | Stopping |
|-------|-------------|--------------|----------|
| 0     | 0.0         | 0.0          | 1000 updates |
| 1     | 1.0         | 0.0          | 1000 updates |
| 2     | 1.0         | 0.6          | early stop (p_miss=0 for 3 steps, 5-step rollback) |

## Installation

```bash
conda create -n rnn-env python=3.10 -y && conda activate rnn-env
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "neurogym @ git+https://github.com/neurogym/neurogym.git"
pip install numpy matplotlib scikit-learn
```

Training runs on CPU by default (faster than GPU for `hidden_size=128`).

## Layout

```
src/
  env.py          CuedTargetWithDistractorsV3
  model.py        BioLeakyRNN (Dale's law, E/I split)
  model_topo.py   topographic variant with 2D spatial sheet
  training.py     TrainConfig, curriculum loop, early stopping
  dataset.py      trial rollout, batch padding
  analysis.py     PCA, dPCA, jPCA, decoding, tangling
  plotting.py     visualisation helpers
notebooks/
  01b_train_v2.ipynb     training (base model)
  01c_train_topo.ipynb   training (topographic)
  02_analysis.ipynb      PCA / dPCA
  03c_jpca_dims.ipynb    jPCA dimensionality
  04_tangling.ipynb      trajectory tangling
  07b_decoding_v2.ipynb  linear decoding
```
