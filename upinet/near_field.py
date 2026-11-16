"""Near-field graph encoder with geometric edge attention and spline messages.

Each update aggregates geometry-conditioned messages from neighbouring nodes and adds their
contribution to the node representation, so after t rounds the state at p_i depends on the
excitation only within t * r_nf of p_i.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .geometry import SurfaceGraph
from .kan import KANMessageKernel


def segment_softmax(x: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    """Softmax of x (E,) or (..., E) over the entries sharing the same index (E,)."""
    shape = x.shape[:-1] + (num,)
    idx = index.expand_as(x)
    xmax = torch.full(shape, float("-inf"), device=x.device, dtype=x.dtype)
    xmax = xmax.scatter_reduce(-1, idx, x, reduce="amax", include_self=True)
    e = torch.exp(x - xmax.gather(-1, idx))
    s = torch.zeros(shape, device=x.device, dtype=x.dtype).scatter_add(-1, idx, e)
    return e / s.gather(-1, idx).clamp_min(1e-30)


def ffn(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class EdgeAttention(nn.Module):
    """b_ij (density), gamma_ij (normals), w_ij = MLP(b_ij (+) gamma_ij)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.F_d, self.F_n, self.mlp = ffn(1, hidden, 1), ffn(3, hidden, 1), ffn(2, hidden, 1)

    def forward(self, g: SurfaceGraph) -> torch.Tensor:
        b = segment_softmax(self.F_d(g.inv_density[g.src, None]).squeeze(-1), g.dst, g.num_nodes)
        gamma = segment_softmax(self.F_n(g.dn).squeeze(-1), g.dst, g.num_nodes)
        return self.mlp(torch.stack([b, gamma], -1)).squeeze(-1)          # (E,)


class NearFieldEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 128, iterations: int = 3,
                 kan_widths=(64, 64), grid_size: int = 5, attn_hidden: int = 64,
                 edge_chunk: int = 0):
        super().__init__()
        self.iterations, self.edge_chunk = iterations, edge_chunk
        self.lift = nn.Linear(in_channels, hidden)                         # J^{(0)} -> node state
        self.edge_attention = EdgeAttention(attn_hidden)
        d_xi = 3 + 3 + 1 + hidden                                          # [dp~, dn, w, J^{(t)}(p_j)]
        self.kernels = nn.ModuleList(KANMessageKernel(d_xi, kan_widths, hidden, grid_size)
                                     for _ in range(iterations))

    @staticmethod
    def _tail(phi, z_geo, z_state_j):
        return phi.tail(z_geo + z_state_j)

    def forward(self, x0: torch.Tensor, g: SurfaceGraph) -> torch.Tensor:
        """x0: (B, K, in_channels) sampled tangential incident field -> J_near state (B, K, hidden)."""
        J = self.lift(x0)
        w = self.edge_attention(g)
        rho = segment_softmax(w, g.dst, g.num_nodes)
        geo = torch.cat([g.dp, g.dn, w[:, None]], -1)                       # (E, 7) static part of xi_ij
        n_geo = geo.shape[-1]
        E = g.src.numel()
        chunk = self.edge_chunk if self.edge_chunk > 0 else E
        for phi in self.kernels:
            first = phi.layers[0]
            # Split the first spline layer over the descriptor [dp~, dn, w | J^{(t)}(p_j)].
            z_geo = first.forward_partial(geo, 0, n_geo)[None]               # per edge, shared by the batch
            z_state = first.forward_partial(J, n_geo, first.in_features)     # per node j
            agg = torch.zeros_like(J)
            for s in range(0, E, chunk):
                sl = slice(s, s + chunk)
                zs = z_state[:, g.src[sl]]
                if self.training and chunk < E:
                    m = checkpoint(self._tail, phi, z_geo[:, sl], zs, use_reentrant=False)
                else:
                    m = self._tail(phi, z_geo[:, sl], zs)
                agg = agg.index_add(1, g.dst[sl], rho[sl, None] * m)
            J = J + agg                                                    # residual update
        return J
