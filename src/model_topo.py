"""
BioLeakyRNNTopo — leaky RNN with E+I on a 2D sheet and a geometric Gaussian
receptive field for spatial inputs (Chen & Gong 2022, Eq. 35).

Sheet layout
------------
- N_E excitatory neurons placed on a regular integer grid (sheet_side × sheet_side),
  then rescaled to normalized coordinates in [-1, +1]^2.
- N_I inhibitory neurons placed at uniformly random continuous positions in the
  same [-1, +1]^2 plane (matching the Chen & Gong 2022 layout, p. 14:
  "N_E excitatory neurons are located at integer coordinates and N_I inhibitory
  neurons are uniformly randomly distributed on the plane").
- exc_ratio = N_E / (N_E + N_I). With sheet_side=12 and exc_ratio=0.80 this yields
  144 E + 36 I = 180 neurons, matching the paper's 4:1 E/I ratio.

Input channels (input_size=7)
-----------------------------
    [0] fixation      — scalar, learnable W_in_fix
    [1] cue_x         — cue x-coordinate (raw)
    [2] cue_y         — cue y-coordinate (raw)
    [3] cue_strength  — cue amplitude (0 when absent)
    [4] stim_x        — stimulus x-coordinate
    [5] stim_y        — stimulus y-coordinate
    [6] stim_strength — stimulus amplitude (0 when absent)

Channels 1-3 and 4-6 are consumed by a purely geometric Gaussian receptive field
(no learnable weights): each neuron i at position r_i receives
    drive_k(t) = strength_k(t) * exp(-|r_i - (x_k(t), y_k(t))|^2 / (2 rf_sigma^2))
Total preact contribution = drive_fix + drive_cue + drive_stim + W_rec h + b + noise.

Recurrent connectivity
----------------------
Distance-dependent probabilistic mask (same form as Chen & Gong Eq.6):
    p_ij = exp(-D_ij / tau_class(i,j)),
where tau_* are expressed in the SAME coordinate units as the sheet — i.e.,
lengths in [-1, +1]^2. Defaults (tau_ee=0.25, tau_ie=0.32, tau_ei=0.64,
tau_ii=0.64) approximate the paper's ratios (D_EE=59.2um, D_IE=74um,
D_EI=D_II=148um on a ~466um sheet).
"""

import math
from typing import Optional

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


class BioLeakyRNNTopo(nn.Module):

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 180,
        output_size: int = 2,
        dt: float = 20.0,
        tau: float = 100.0,
        activation: str = "softplus",
        sigma_rec: float = 0.05,
        rec_init: str = "diag",
        rec_scale: Optional[float] = None,
        batch_first: bool = True,
        learn_h0: bool = False,
        use_ei: bool = True,
        exc_ratio: float = 0.80,
        use_dale: bool = True,
        rec_sparsity: float = 0.0,
        out_sparsity: float = 0.0,
        allow_self_connections: bool = True,
        mask_seed: Optional[int] = None,
        dale_on_output: bool = False,
        sheet_side: int = 12,
        tau_ee: float = 0.25,
        tau_ie: float = 0.32,
        tau_ei: float = 0.64,
        tau_ii: float = 0.64,
        rf_sigma: float = 0.3,
        tau_e_range: Optional[tuple] = None,
        tau_i_range: Optional[tuple] = None,
        use_adaptation: bool = False,
        tau_adapt: float = 200.0,
        g_adapt: float = 0.0,
    ):
        super().__init__()

        use_hetero_tau = (tau_e_range is not None) or (tau_i_range is not None)
        if not use_hetero_tau:
            alpha = dt / tau
            if not (0.0 < alpha <= 1.0):
                raise ValueError(f"alpha = dt/tau must be in (0, 1], got {alpha:.4f}")
        if not (0.0 < exc_ratio < 1.0):
            raise ValueError(f"exc_ratio must be in (0,1), got {exc_ratio}")
        for pname, s in [("rec_sparsity", rec_sparsity), ("out_sparsity", out_sparsity)]:
            if not (0.0 <= s < 1.0):
                raise ValueError(f"{pname} must be in [0,1), got {s}")
        if input_size != 7:
            raise ValueError(
                f"BioLeakyRNNTopo expects input_size=7 (fix + cue XYS + stim XYS), got {input_size}"
            )

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
        self.out_sparsity = float(out_sparsity)
        self.allow_self_connections = bool(allow_self_connections)
        self.dale_on_output = bool(dale_on_output)
        self.sheet_side = int(sheet_side)
        self.rf_sigma = float(rf_sigma)

        # Optional Chen & Gong 2022-style neural adaptation. This is the rate
        # analogue of slow K+ current driving spike-frequency adaptation:
        #   da_i/dt = -a_i / tau_adapt + g_adapt * h_i
        # then preact_i <- preact_i - a_i. NOT Tsodyks-Markram synaptic
        # depression — that would scale recurrent input by per-presynaptic-
        # neuron resources s_i instead.
        self.use_adaptation = bool(use_adaptation)
        self.tau_adapt = float(tau_adapt)
        self.g_adapt = float(g_adapt)
        if self.use_adaptation:
            adapt_decay = math.exp(-float(dt) / self.tau_adapt)
            self.register_buffer("adapt_decay",
                                 torch.tensor(adapt_decay, dtype=torch.float32))
        self._a_state = None  # runtime, not in state_dict; reset per forward()

        if rec_scale is None:
            rec_scale = 1.0 if activation == "tanh" else 0.8
        self.rec_scale = float(rec_scale)

        self.sigma_rec = float(sigma_rec)

        # n_exc is determined by the sheet geometry, not exc_ratio
        n_exc = self.sheet_side * self.sheet_side
        n_inh = max(1, hidden_size - n_exc)
        if n_exc + n_inh != hidden_size:
            raise ValueError(
                f"hidden_size={hidden_size} incompatible with sheet_side={sheet_side} "
                f"(need n_exc + n_inh = hidden_size; got {n_exc}+{n_inh}={n_exc+n_inh})"
            )
        actual_ratio = n_exc / hidden_size
        if abs(actual_ratio - self.exc_ratio) > 0.05:
            self.exc_ratio = actual_ratio
        self.n_exc = n_exc
        self.n_inh = n_inh

        ei_sign = torch.ones(hidden_size, dtype=torch.float32)
        if self.use_ei:
            ei_sign[n_exc:] = -1.0
        self.register_buffer("ei_sign", ei_sign)

        # Per-neuron membrane time constants (Murray-style heterogeneity).
        # When both ranges are None: scalar `tau` is used and `alpha` stays a
        # 0-d buffer — identical to the original homogeneous behaviour, so old
        # checkpoints still load. When either range is given: sample log-uniform
        # tau per neuron (E first, then I, matching the layout in `coords`) and
        # use exponential-Euler decay so taus shorter than `dt` stay stable
        # (forward-Euler would require tau >= dt).
        if use_hetero_tau:
            gen_tau = torch.Generator()
            gen_tau.manual_seed((mask_seed or 0) + 7)
            tau_per = torch.full((hidden_size,), float(tau), dtype=torch.float32)
            if tau_e_range is not None:
                lo, hi = float(tau_e_range[0]), float(tau_e_range[1])
                u = torch.rand(n_exc, generator=gen_tau)
                tau_per[:n_exc] = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            if tau_i_range is not None:
                lo, hi = float(tau_i_range[0]), float(tau_i_range[1])
                u = torch.rand(n_inh, generator=gen_tau)
                tau_per[n_exc:] = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            alpha_per = 1.0 - torch.exp(-float(dt) / tau_per)  # exp-Euler decay
            self.register_buffer("alpha", alpha_per)
            self.register_buffer("tau_per_neuron", tau_per)
            sigma_eff_per = torch.sqrt(2.0 / alpha_per) * self.sigma_rec
            self.register_buffer("sigma_eff", sigma_eff_per.to(torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            self.register_buffer(
                "sigma_eff",
                torch.tensor(math.sqrt(2.0 / alpha) * self.sigma_rec, dtype=torch.float32),
            )

        gen = None
        if mask_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(mask_seed)

        coords = self._make_sheet_coords(
            n_exc=n_exc, n_inh=n_inh, sheet_side=self.sheet_side, seed=mask_seed,
        )  # [H, 2]
        self.register_buffer("coords", coords)

        rec_mask = self._make_sheet_mask(
            coords=coords, n_exc=n_exc, n_inh=n_inh,
            tau_ee=tau_ee, tau_ie=tau_ie, tau_ei=tau_ei, tau_ii=tau_ii,
            seed=mask_seed,
        )
        if not allow_self_connections:
            rec_mask.fill_diagonal_(0.0)
        self.register_buffer("rec_mask", rec_mask)
        self.register_buffer("out_mask", self._make_mask(output_size, hidden_size, out_sparsity, gen))

        # fixation is the only learned input channel; cue/stim channels go through the RF
        self.W_in_fix = nn.Parameter(torch.empty(hidden_size))
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h   = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))

        # auxiliary xy readout — trained with MSE to keep the code spatial/continuous
        # (no Dale constraint; not part of the action pathway)
        self.W_aux = nn.Parameter(torch.empty(2, hidden_size))
        self.b_aux = nn.Parameter(torch.zeros(2))

        if learn_h0:
            self.h0 = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("h0", torch.zeros(hidden_size))

        self.noise_at_eval = False

        self.rec_init = rec_init
        self.reset_parameters()

    @staticmethod
    def _make_mask(rows, cols, sparsity, generator=None):
        if sparsity <= 0.0:
            return torch.ones(rows, cols, dtype=torch.float32)
        return (torch.rand(rows, cols, generator=generator) < (1.0 - sparsity)).float()

    @staticmethod
    def _make_sheet_coords(n_exc: int, n_inh: int, sheet_side: int,
                           seed: Optional[int] = None) -> torch.Tensor:
        """E on integer grid -> [-1,+1]^2, I uniform random. Returns [H, 2], E first."""
        assert sheet_side * sheet_side == n_exc, \
            f"n_exc={n_exc} must equal sheet_side^2 = {sheet_side*sheet_side}"

        idx = torch.arange(n_exc)
        xi = (idx % sheet_side).float()
        yi = (idx // sheet_side).float()
        denom = max(1.0, float(sheet_side - 1))
        e_coords = torch.stack([xi / denom * 2.0 - 1.0, yi / denom * 2.0 - 1.0], dim=1)

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        i_coords = torch.rand(n_inh, 2, generator=gen) * 2.0 - 1.0

        return torch.cat([e_coords, i_coords], dim=0)

    @staticmethod
    def _make_sheet_mask(coords: torch.Tensor,
                         n_exc: int, n_inh: int,
                         tau_ee: float, tau_ie: float,
                         tau_ei: float, tau_ii: float,
                         seed: Optional[int] = None) -> torch.Tensor:
        """
        Distance-dependent recurrent mask on a TOROIDAL sheet (period 2.0 in
        normalized [-1, +1] coordinates). Edge neurons therefore have the same
        local connectivity as central ones. tau_* are lengths expressed in the
        same coordinate units as `coords`. Convention: mask[i, j] is the
        connection post-i <- pre-j.
        """
        n_tot = n_exc + n_inh
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)   # [H, H, 2]
        # Toroidal wrap to shortest periodic image (period 2.0 -> diff in [-1, +1])
        diff = diff - 2.0 * torch.round(diff / 2.0)
        D = diff.norm(dim=-1)                              # [H, H]

        tau_mat = torch.zeros(n_tot, n_tot)
        tau_mat[:n_exc, :n_exc] = tau_ee   # post E <- pre E
        tau_mat[n_exc:, :n_exc] = tau_ie   # post I <- pre E
        tau_mat[:n_exc, n_exc:] = tau_ei   # post E <- pre I
        tau_mat[n_exc:, n_exc:] = tau_ii   # post I <- pre I

        prob = torch.exp(-D / tau_mat)

        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed + 1)
        mask = (torch.rand(n_tot, n_tot, generator=gen) < prob).float()
        return mask

    def reset_parameters(self):
        with torch.no_grad():
            self.W_in_fix.normal_(0.0, 1.0 / math.sqrt(self.input_size))

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
            self.W_aux.normal_(0.0, 1.0 / math.sqrt(self.hidden_size))

        nn.init.zeros_(self.b_h)
        nn.init.zeros_(self.b_out)
        nn.init.zeros_(self.b_aux)
        if isinstance(self.h0, nn.Parameter):
            nn.init.zeros_(self.h0)

        if self.use_dale:
            with torch.no_grad():
                self.W_rec.abs_()
                if self.dale_on_output:
                    self.W_out.abs_()

        with torch.no_grad():
            self.W_rec.mul_(self.rec_mask)
            self.W_out.mul_(self.out_mask)

    def init_hidden(self, batch_size, device):
        # Also initialize adaptation state so callers that loop over step()
        # directly (e.g. ablation rollouts in 05_spatial_analysis_topo) get
        # consistent adaptation behaviour without going through forward().
        if self.use_adaptation:
            self._a_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)

    def _gaussian_rf_drive(self, xy_s):
        """xy_s: [..., 3] = (x, y, strength) -> [..., H] drive per neuron.

        Distance is computed on the toroidal sheet (period 2.0 in [-1, +1]),
        consistent with `_make_sheet_mask`. A stimulus near one edge therefore
        also drives neurons near the opposite edge via the wrap.
        """
        xy = xy_s[..., :2]
        s  = xy_s[..., 2:3]
        diff = self.coords.view(*([1] * (xy.dim() - 1)), self.hidden_size, 2) \
             - xy.unsqueeze(-2)
        # Toroidal wrap to shortest periodic image
        diff = diff - 2.0 * torch.round(diff / 2.0)
        d2 = diff.pow(2).sum(dim=-1)
        return s * torch.exp(-d2 / (2.0 * self.rf_sigma * self.rf_sigma))

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

    def decode_xy(self, h):
        """h [..., H] -> predicted (x, y) [..., 2] via the auxiliary readout."""
        return F.linear(h, self.W_aux, self.b_aux)

    def step(self, x_t, h_t):
        fix  = x_t[..., 0:1]   # fixation channel
        cue  = x_t[..., 1:4]   # cue (x, y, strength)
        stim = x_t[..., 4:7]   # stim (x, y, strength)

        drive_fix  = fix * self.W_in_fix.view(1, -1)
        drive_cue  = self._gaussian_rf_drive(cue)
        drive_stim = self._gaussian_rf_drive(stim)

        preact = drive_fix + drive_cue + drive_stim \
               + F.linear(h_t, self.effective_W_rec()) + self.b_h

        # Neural adaptation hyperpolarizes pre-activation (Chen & Gong style).
        # _a_state is managed by forward(); when called from forward it is set
        # for the right batch size, so just subtract it here.
        if self.use_adaptation and self._a_state is not None:
            preact = preact - self._a_state

        if (self.training or self.noise_at_eval) and self.sigma_rec > 0.0:
            preact = preact + self.sigma_eff * torch.randn_like(preact)

        h_next = (1.0 - self.alpha) * h_t + self.alpha * self.phi(preact)

        # Exp-Euler update of adaptation: a builds with activity, decays with tau_adapt
        if self.use_adaptation and self._a_state is not None:
            self._a_state = (
                self.adapt_decay * self._a_state
                + (1.0 - self.adapt_decay) * self.g_adapt * h_next
            )

        y_t = F.linear(h_next, self.effective_W_out(), self.b_out)
        return h_next, y_t

    def forward(self, x, h0=None, return_hidden=False):
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

        # Reset adaptation state to zero per forward pass (per-batch).
        if self.use_adaptation:
            self._a_state = torch.zeros(B, self.hidden_size, device=x.device)

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
