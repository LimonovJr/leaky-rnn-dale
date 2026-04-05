"""
BioLeakyRNN — leaky RNN with E/I split, Dale's law, sparse masks, recurrent noise.

Dynamics:
    h_{t+1} = (1 - alpha) * h_t + alpha * phi(W_rec_eff h_t + W_in_eff x_t + b_h + noise)
    y_t     = W_out_eff h_t + b_out

    alpha   = dt / tau
    noise   ~ N(0, sigma_eff^2),  sigma_eff = sqrt(2/alpha) * sigma_rec
    (Mante et al. / Perez-Nieves scaling: keeps steady-state noise variance
     independent of dt/tau)

Dale's law:
    F.linear(x, W) = x @ W.T, so preact_j = sum_i h_i * W_rec[j,i].
    Neuron i is the sender -> sign constraint on column i of W_rec.
    W_rec stores magnitudes; ei_sign vector applies ±1 per sender column.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = name.lower()
    if name == "softplus": return F.softplus
    if name == "tanh":     return torch.tanh
    if name == "relu":     return F.relu
    if name == "retanh":   return lambda x: torch.tanh(F.relu(x))
    raise ValueError(f"Unknown activation: {name!r}")


class BioLeakyRNN(nn.Module):

    def __init__(
        self,
        input_size: int = 7,       # V3 obs dim
        hidden_size: int = 128,
        output_size: int = 2,
        dt: float = 20.0,
        tau: float = 100.0,
        activation: str = "softplus",
        sigma_rec: float = 0.05,
        rec_init: str = "diag",    # "diag" | "randgauss"
        rec_scale: Optional[float] = None,
        batch_first: bool = True,
        learn_h0: bool = False,
        use_ei: bool = True,
        exc_ratio: float = 0.8,
        use_dale: bool = True,
        rec_sparsity: float = 0.0,
        in_sparsity: float = 0.0,
        out_sparsity: float = 0.0,
        allow_self_connections: bool = True,
        mask_seed: Optional[int] = None,
        dale_on_output: bool = False,
    ):
        super().__init__()

        alpha = dt / tau
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha = dt/tau must be in (0, 1], got {alpha:.4f}")
        if not (0.0 < exc_ratio < 1.0):
            raise ValueError(f"exc_ratio must be in (0,1), got {exc_ratio}")
        for pname, s in [("rec_sparsity", rec_sparsity), ("in_sparsity", in_sparsity),
                         ("out_sparsity", out_sparsity)]:
            if not (0.0 <= s < 1.0):
                raise ValueError(f"{pname} must be in [0,1), got {s}")

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt          = float(dt)
        self.tau         = float(tau)
        self.batch_first = batch_first
        self.activation_name = activation
        self.phi = get_activation(activation)

        self.use_ei    = bool(use_ei)
        self.exc_ratio = float(exc_ratio)
        self.use_dale  = bool(use_dale)
        self.rec_sparsity = float(rec_sparsity)
        self.in_sparsity  = float(in_sparsity)
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
            torch.tensor(math.sqrt(2.0 / alpha) * self.sigma_rec, dtype=torch.float32)
        )

        # E/I partition
        n_exc = max(1, min(hidden_size - 1, int(round(hidden_size * exc_ratio))))
        self.n_exc = n_exc
        self.n_inh = hidden_size - n_exc

        ei_sign = torch.ones(hidden_size, dtype=torch.float32)
        if self.use_ei:
            ei_sign[n_exc:] = -1.0
        self.register_buffer("ei_sign", ei_sign)

        # connectivity masks
        gen = None
        if mask_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(mask_seed)

        rec_mask = self._make_mask(hidden_size, hidden_size, rec_sparsity, gen)
        if not allow_self_connections:
            rec_mask.fill_diagonal_(0.0)
        self.register_buffer("rec_mask", rec_mask)
        self.register_buffer("in_mask",  self._make_mask(hidden_size, input_size,  in_sparsity,  gen))
        self.register_buffer("out_mask", self._make_mask(output_size, hidden_size, out_sparsity, gen))

        # raw parameters (Dale: magnitudes only; non-Dale: signed weights)
        self.W_in  = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h   = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))

        if learn_h0:
            self.h0 = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("h0", torch.zeros(hidden_size))

        self.rec_init = rec_init
        self.reset_parameters()

    @staticmethod
    def _make_mask(rows, cols, sparsity, generator=None):
        if sparsity <= 0.0:
            return torch.ones(rows, cols, dtype=torch.float32)
        return (torch.rand(rows, cols, generator=generator) < (1.0 - sparsity)).float()

    def reset_parameters(self):
        with torch.no_grad():
            self.W_in.normal_(0.0, 1.0 / math.sqrt(self.input_size))

            if self.rec_init == "diag":
                self.W_rec.zero_()
                self.W_rec.add_(torch.diag(self.rec_scale * torch.ones(self.hidden_size,
                                                                        device=self.W_rec.device)))
                self.W_rec.add_(0.01 * torch.randn_like(self.W_rec))
            elif self.rec_init == "randgauss":
                self.W_rec.normal_(0.0, self.rec_scale / math.sqrt(self.hidden_size))
            else:
                raise ValueError(f"Unknown rec_init: {self.rec_init!r}")

            self.W_out.normal_(0.0, 1.0 / math.sqrt(self.hidden_size))

        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_out)
        if isinstance(self.h0, nn.Parameter):
            nn.init.zeros_(self.h0)

        if self.use_dale:
            with torch.no_grad():
                self.W_rec.abs_()
                if self.dale_on_output:
                    self.W_out.abs_()

        with torch.no_grad():
            self.W_in.mul_(self.in_mask)
            self.W_rec.mul_(self.rec_mask)
            self.W_out.mul_(self.out_mask)

    def init_hidden(self, batch_size, device):
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)

    def effective_W_in(self):
        return self.W_in * self.in_mask

    def effective_W_rec(self):
        W = self.W_rec * self.rec_mask
        if self.use_dale:
            W = W.abs() * self.ei_sign.view(1, -1)
        return W

    def effective_W_out(self):
        W = self.W_out * self.out_mask
        if self.use_dale and self.dale_on_output:
            W = W.abs() * self.ei_sign.view(1, -1)
        return W

    def step(self, x_t, h_t):
        """x_t: [B, D], h_t: [B, H] -> h_next: [B, H], y_t: [B, C]"""
        preact = F.linear(x_t, self.effective_W_in()) + \
                 F.linear(h_t, self.effective_W_rec()) + self.b_h

        if self.training and self.sigma_rec > 0.0:
            preact = preact + self.sigma_eff * torch.randn_like(preact)

        h_next = (1.0 - self.alpha) * h_t + self.alpha * self.phi(preact)
        y_t = F.linear(h_next, self.effective_W_out(), self.b_out)
        return h_next, y_t

    def forward(self, x, h0=None, return_hidden=False):
        """
        x: [B, T, D] (batch_first=True) or [T, B, D]
        returns: logits [B, T, C], h_last [B, H], (h_seq [B, T, H] if return_hidden)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be 3-D, got {tuple(x.shape)}")
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, D = x.shape
        if D != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {D}")

        h = self.init_hidden(B, x.device) if h0 is None else h0
        if h0 is not None and h0.shape != (B, self.hidden_size):
            raise ValueError(f"h0 shape mismatch: expected {(B, self.hidden_size)}, got {tuple(h0.shape)}")

        logits_list = []
        hidden_list = [] if return_hidden else None

        for t in range(T):
            h, y_t = self.step(x[:, t, :], h)
            logits_list.append(y_t)
            if return_hidden:
                hidden_list.append(h)

        logits = torch.stack(logits_list, dim=1)
        if not self.batch_first:
            logits = logits.transpose(0, 1)

        if return_hidden:
            h_seq = torch.stack(hidden_list, dim=1)
            if not self.batch_first:
                h_seq = h_seq.transpose(0, 1)
            return logits, h, h_seq

        return logits, h
