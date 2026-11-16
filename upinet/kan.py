"""Learnable univariate-basis (Kolmogorov-Arnold) layers.

z_c = sum_l psi_lc(x_l),  psi_lc(x) = w^base_lc * silu(x) + sum_b w^spline_lcb * B_b(x),

with cubic B-splines B_b on a uniform grid of `grid_size` intervals over `grid_range`.
The SiLU base term contributes outside the spline support. Uniform cubic B-splines are
translates of the cardinal cubic B-spline, evaluated here in closed form.

Because a layer is a SUM of univariate functions of its inputs, it can be evaluated on any
subset of inputs and the partial sums added (`forward_partial`). The near-field encoder uses
this to evaluate the neighbour-state part of the first layer once per node instead of once
per edge.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cardinal_cubic_bspline(s: torch.Tensor) -> torch.Tensor:
    """Cardinal cubic B-spline supported on [0, 4]."""
    r = (s - 2.0).abs()
    inner = 2.0 / 3.0 - r ** 2 + 0.5 * r ** 3
    outer = (2.0 - r).clamp_min(0.0) ** 3 / 6.0
    return torch.where(r < 1.0, inner, outer)


def cardinal_cubic_bspline_deriv(s: torch.Tensor) -> torch.Tensor:
    d = s - 2.0
    r = d.abs()
    inner = -2.0 * r + 1.5 * r ** 2
    outer = -0.5 * (2.0 - r).clamp_min(0.0) ** 2
    return torch.where(r < 1.0, inner, outer) * torch.sign(d)


class _SplineSum(torch.autograd.Function):
    """out[..., o] = sum_{i,b} B_b(x[..., i]) W[o, i, b]; saves only x and W (bases recomputed)."""

    @staticmethod
    def forward(ctx, x, W, t0, h):
        ctx.save_for_backward(x, W, t0)
        ctx.h = h
        with torch.no_grad():
            B = cardinal_cubic_bspline((x.unsqueeze(-1) - t0) / h)
            return torch.einsum("...ib,oib->...o", B, W)

    @staticmethod
    def backward(ctx, g):
        x, W, t0 = ctx.saved_tensors
        s = (x.unsqueeze(-1) - t0) / ctx.h
        gx = gW = None
        if ctx.needs_input_grad[0]:
            tmp = torch.einsum("...o,oib->...ib", g, W)
            gx = (tmp * cardinal_cubic_bspline_deriv(s)).sum(-1) / ctx.h
        if ctx.needs_input_grad[1]:
            B = cardinal_cubic_bspline(s)
            gW = torch.einsum("...o,...ib->oib", g.reshape(-1, g.shape[-1]),
                              B.reshape(-1, *B.shape[-2:]))
        return gx, gW, None, None


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                 grid_range=(-2.0, 2.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        h = (grid_range[1] - grid_range[0]) / grid_size
        n_basis = grid_size + 3
        self.h = h
        self.register_buffer("t0", torch.arange(n_basis, dtype=torch.float32) * h + grid_range[0] - 3 * h)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(in_features))

    def bsplines(self, x: torch.Tensor) -> torch.Tensor:
        """(..., in) -> (..., in, n_basis)."""
        return cardinal_cubic_bspline((x.unsqueeze(-1) - self.t0) / self.h)

    def forward_partial(self, x: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Contribution of inputs start:stop (x has stop-start features)."""
        base = F.linear(F.silu(x), self.base_weight[:, start:stop])
        spline = _SplineSum.apply(x, self.spline_weight[:, start:stop], self.t0, self.h)
        return base + spline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_partial(x, 0, self.in_features)


class KANMessageKernel(nn.Module):
    """Phi^{(t)}: R spline layers followed by W_o."""

    def __init__(self, in_features: int, widths, out_features: int, grid_size: int = 5):
        super().__init__()
        dims = [in_features] + list(widths)
        self.layers = nn.ModuleList(KANLayer(a, b, grid_size) for a, b in zip(dims[:-1], dims[1:]))
        self.W_o = nn.Linear(dims[-1], out_features, bias=False)

    def tail(self, z: torch.Tensor) -> torch.Tensor:
        """Layers 2..R and W_o applied to the first-layer output z."""
        for layer in self.layers[1:]:
            z = layer(z)
        return self.W_o(z)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.tail(self.layers[0](xi))
