"""
BioLeakyRNN — leaky RNN with optional E/I split, Dale's law,
sparse connectivity masks, and article-style recurrent noise.

Dynamics:
    h_{t+1} = (1 - alpha) * h_t + alpha * phi(W_rec_eff @ h_t + W_in_eff @ x_t + b_h + noise)

Readout:
    y_t = W_out_eff @ h_t + b_out

Notes on Dale's law
-------------------
F.linear(x, W) computes x @ W.T, so the recurrent preactivation of neuron j is:
    preact_j = sum_i h_i * W_rec[j, i]
Neuron i is the "sender", so Dale's law constrains the sign of *column* i of W_rec:
  - excitatory neuron i  -> W_rec[:, i] >= 0
  - inhibitory neuron i  -> W_rec[:, i] <= 0

sigma_eff scaling
-----------------
Following Mante et al. / Perez-Nieves et al. convention:
    sigma_eff = sqrt(2 / alpha) * sigma_rec
This keeps the steady-state noise variance independent of dt/tau.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = name.lower()
    if name == "softplus":
        return F.softplus
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return F.relu
    if name == "retanh":
        return lambda x: torch.tanh(F.relu(x))
    raise ValueError(f"Unknown activation: {name!r}")


class BioLeakyRNN(nn.Module):
    """
    Leaky RNN with biologically-inspired constraints.

    Parameters
    ----------
    input_size : int
    hidden_size : int
    output_size : int
    dt : float
        Simulation timestep in ms.
    tau : float
        Membrane time constant in ms.
    activation : str
        One of 'softplus', 'tanh', 'relu', 'retanh'.
    sigma_rec : float
        Recurrent noise amplitude (scaled internally to sigma_eff).
    rec_init : str
        'diag' — diagonal + small noise; 'randgauss' — Gaussian.
    rec_scale : float or None
        Scale of initial recurrent weights. Defaults to 1.0 (tanh) or 0.8 (others).
    batch_first : bool
    learn_h0 : bool
        If True, initial hidden state is a learnable parameter.
    use_ei : bool
        Split neurons into excitatory / inhibitory populations.
    exc_ratio : float
        Fraction of excitatory neurons (only used when use_ei=True).
    use_dale : bool
        Enforce Dale's law on recurrent (and optionally output) weights.
    rec_sparsity : float
        Fraction of recurrent weights permanently set to zero.
    in_sparsity : float
        Fraction of input weights permanently set to zero.
    out_sparsity : float
        Fraction of output weights permanently set to zero.
    allow_self_connections : bool
    mask_seed : int or None
        Random seed for connectivity masks.
    dale_on_output : bool
        If True, apply Dale's law sign constraints to output weights as well.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 128,
        output_size: int = 2,
        dt: float = 20.0,
        tau: float = 100.0,
        activation: str = "softplus",
        sigma_rec: float = 0.05,
        rec_init: str = "diag",
        rec_scale: Optional[float] = None,
        batch_first: bool = True,
        learn_h0: bool = False,
        # biologically-inspired options
        use_ei: bool = True,
        exc_ratio: float = 0.8,
        use_dale: bool = True,
        # fixed sparsity fractions in [0, 1)
        rec_sparsity: float = 0.0,
        in_sparsity: float = 0.0,
        out_sparsity: float = 0.0,
        # mask / connectivity options
        allow_self_connections: bool = True,
        mask_seed: Optional[int] = None,
        # Dale on output readout
        dale_on_output: bool = False,
    ):
        super().__init__()

        alpha = dt / tau
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha = dt/tau must be in (0, 1], got {alpha:.4f}")
        if not (0.0 < exc_ratio < 1.0):
            raise ValueError(f"exc_ratio must be in (0, 1), got {exc_ratio}")
        for param_name, s in [
            ("rec_sparsity", rec_sparsity),
            ("in_sparsity", in_sparsity),
            ("out_sparsity", out_sparsity),
        ]:
            if not (0.0 <= s < 1.0):
                raise ValueError(f"{param_name} must be in [0, 1), got {s}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = float(dt)
        self.tau = float(tau)
        self.batch_first = batch_first
        self.activation_name = activation
        self.phi = get_activation(activation)

        self.use_ei = bool(use_ei)
        self.exc_ratio = float(exc_ratio)
        self.use_dale = bool(use_dale)
        self.rec_sparsity = float(rec_sparsity)
        self.in_sparsity = float(in_sparsity)
        self.out_sparsity = float(out_sparsity)
        self.allow_self_connections = bool(allow_self_connections)
        self.dale_on_output = bool(dale_on_output)

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        if rec_scale is None:
            rec_scale = 1.0 if activation == "tanh" else 0.8
        self.rec_scale = float(rec_scale)

        self.sigma_rec = float(sigma_rec)
        self.register_buffer(
            "sigma_eff",
            torch.tensor(math.sqrt(2.0 / alpha) * self.sigma_rec, dtype=torch.float32),
        )

        # E/I partition
        n_exc = int(round(hidden_size * exc_ratio))
        n_exc = max(1, min(hidden_size - 1, n_exc))
        self.n_exc = n_exc
        self.n_inh = hidden_size - n_exc

        ei_sign = torch.ones(hidden_size, dtype=torch.float32)
        if self.use_ei:
            ei_sign[n_exc:] = -1.0
        self.register_buffer("ei_sign", ei_sign)

        # connectivity masks
        generator = None
        if mask_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(mask_seed)

        rec_mask = self._make_mask(hidden_size, hidden_size, rec_sparsity, generator)
        if not allow_self_connections:
            rec_mask.fill_diagonal_(0.0)
        self.register_buffer("rec_mask", rec_mask)

        in_mask = self._make_mask(hidden_size, input_size, in_sparsity, generator)
        self.register_buffer("in_mask", in_mask)

        out_mask = self._make_mask(output_size, hidden_size, out_sparsity, generator)
        self.register_buffer("out_mask", out_mask)

        # raw trainable parameters
        # When use_dale=True, W_rec / W_out store *magnitudes* (abs-valued);
        # signs are applied via ei_sign in effective_W_rec / effective_W_out.
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))

        if learn_h0:
            self.h0 = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("h0", torch.zeros(hidden_size))

        self.rec_init = rec_init
        self.reset_parameters()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mask(
        rows: int, cols: int, sparsity: float, generator=None
    ) -> torch.Tensor:
        if sparsity <= 0.0:
            return torch.ones(rows, cols, dtype=torch.float32)
        keep_prob = 1.0 - sparsity
        return (torch.rand(rows, cols, generator=generator) < keep_prob).float()

    def reset_parameters(self):
        with torch.no_grad():
            # input weights
            self.W_in.normal_(mean=0.0, std=1.0 / math.sqrt(self.input_size))

            # recurrent weights
            if self.rec_init == "diag":
                self.W_rec.zero_()
                self.W_rec.add_(
                    torch.diag(
                        self.rec_scale * torch.ones(self.hidden_size, device=self.W_rec.device)
                    )
                )
                self.W_rec.add_(0.01 * torch.randn_like(self.W_rec))
            elif self.rec_init == "randgauss":
                self.W_rec.normal_(
                    mean=0.0, std=self.rec_scale / math.sqrt(self.hidden_size)
                )
            else:
                raise ValueError(f"Unknown rec_init: {self.rec_init!r}")

            self.W_out.normal_(mean=0.0, std=1.0 / math.sqrt(self.hidden_size))

        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_out)
        if isinstance(self.h0, nn.Parameter):
            nn.init.zeros_(self.h0)

        # If using Dale, store magnitudes so the sign transform is stable from the start
        if self.use_dale:
            with torch.no_grad():
                self.W_rec.abs_()
                if self.dale_on_output:
                    self.W_out.abs_()

        # zero out masked connections
        with torch.no_grad():
            self.W_in.mul_(self.in_mask)
            self.W_rec.mul_(self.rec_mask)
            self.W_out.mul_(self.out_mask)

    # ------------------------------------------------------------------
    # effective weights (apply masks + Dale signs)
    # ------------------------------------------------------------------

    def effective_W_in(self) -> torch.Tensor:
        return self.W_in * self.in_mask

    def effective_W_rec(self) -> torch.Tensor:
        W = self.W_rec * self.rec_mask
        if self.use_dale:
            W = W.abs() * self.ei_sign.view(1, -1)
        return W

    def effective_W_out(self) -> torch.Tensor:
        W = self.W_out * self.out_mask
        if self.use_dale and self.dale_on_output:
            W = W.abs() * self.ei_sign.view(1, -1)
        return W

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)

    def step(
        self, x_t: torch.Tensor, h_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep update.

        Parameters
        ----------
        x_t : [B, input_size]
        h_t : [B, hidden_size]

        Returns
        -------
        h_next : [B, hidden_size]
        y_t    : [B, output_size]
        """
        W_in_eff = self.effective_W_in()
        W_rec_eff = self.effective_W_rec()
        W_out_eff = self.effective_W_out()

        preact = F.linear(x_t, W_in_eff) + F.linear(h_t, W_rec_eff) + self.b_h

        if self.training and self.sigma_rec > 0.0:
            preact = preact + self.sigma_eff * torch.randn_like(preact)

        h_next = (1.0 - self.alpha) * h_t + self.alpha * self.phi(preact)
        y_t = F.linear(h_next, W_out_eff, self.b_out)
        return h_next, y_t

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """
        Parameters
        ----------
        x : [B, T, D] if batch_first else [T, B, D]
        h0 : [B, hidden_size] or None
        return_hidden : bool
            If True, also return the full hidden-state sequence.

        Returns
        -------
        logits  : [B, T, output_size]  (or [T, B, output_size] if not batch_first)
        h_last  : [B, hidden_size]
        h_seq   : [B, T, hidden_size]  (only when return_hidden=True)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}")

        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, D = x.shape
        if D != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {D}")

        if h0 is None:
            h = self.init_hidden(B, x.device)
        else:
            if h0.shape != (B, self.hidden_size):
                raise ValueError(
                    f"h0 must have shape {(B, self.hidden_size)}, got {tuple(h0.shape)}"
                )
            h = h0

        logits_list = []
        hidden_list = [] if return_hidden else None

        for t in range(T):
            h, y_t = self.step(x[:, t, :], h)
            logits_list.append(y_t)
            if return_hidden:
                hidden_list.append(h)

        logits = torch.stack(logits_list, dim=1)  # [B, T, C]

        if not self.batch_first:
            logits = logits.transpose(0, 1)

        if return_hidden:
            h_seq = torch.stack(hidden_list, dim=1)  # [B, T, H]
            if not self.batch_first:
                h_seq = h_seq.transpose(0, 1)
            return logits, h, h_seq

        return logits, h
