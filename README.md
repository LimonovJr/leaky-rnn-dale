# Leaky RNN with Dale's Law

Biologically-constrained leaky RNN trained on a cued visuospatial detection task (target + distractors, variable CTOA). Implements E/I split, Dale's law, sparse connectivity, and recurrent noise. Reproduces rotational dynamics reported in Amengual et al.

## Models

**BioLeakyRNN** (`src/model.py`) — base model. Recurrent layer with E/I sign constraints enforced via a fixed `ei_sign` vector on `W_rec`. Trained in three stages: target-only → add cue → add distractors with early stopping.

**BioLeakyRNNTopo** (`src/model_topo.py`) — topographic variant. Neurons arranged on a 2D sheet (12×12 E, random I). Input connections are frozen Gaussian receptive fields; recurrent connectivity falls off with sheet distance. Forces the network to use spatial structure rather than timing shortcuts.

## Notebooks

| | |
|---|---|
| `01b_train_v2` | Train base model (3-stage curriculum) |
| `01c_train_topo` | Train topographic model |
| `02_analysis` | PCA, dPCA, spatial separation — base model |
| `02b_analysis_topo` | Same analysis on topographic checkpoint |
| `03c_jpca_dims` | jPCA: compare 3- vs 6-dim subspace (Amengual vs Churchland) |
| `03d_jpca_dims_topo` | jPCA on topographic checkpoint |
| `04_tangling` | Trajectory tangling by CTOA bin — base model |
| `04b_tangling_topo` | Tangling on topographic checkpoint |
| `05b_oscillations_v2` | Oscillatory modes: W_rec eigenspectrum, Jacobian, Welch PSD, jPCA frequency |
| `05c_oscillations_topo` | Same oscillation analysis on topographic checkpoint |
| `05_spatial_analysis_topo` | Spatial coding: activity maps, linear decoders, ablations |
| `06_stimulus_degradation` | Robustness under degraded cue / target / noise |
| `07b_decoding_v2` | Decode target location across time; correlate tangling with RT |
| `09_topo_diagnostics` | Topographic model diagnostics: lesion tests, selectivity, sheet-map |

## Installation

```bash
conda create -n rnn-env python=3.10 -y && conda activate rnn-env
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "neurogym @ git+https://github.com/neurogym/neurogym.git"
pip install numpy matplotlib scikit-learn
```
