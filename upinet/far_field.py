"""Hierarchical far-field branch with octree pooling, attention, and kernel-point fusion.

The octree runs from level 1 (coarsest) to level L (finest). The near-field state is
pooled from points into the finest cells. Cross-attention uses parent context lifted
to each level, and fusion uses point coordinates normalized within the parent cell.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import Octree


def scatter_mean_max(x: torch.Tensor, index: torch.Tensor, num: int):
    """x (B, n, d) grouped by index (n,) -> mean, max (B, num, d)."""
    B, _, d = x.shape
    idx = index[None, :, None].expand(B, -1, d)
    s = torch.zeros(B, num, d, device=x.device, dtype=x.dtype).scatter_add(1, idx, x)
    cnt = torch.zeros(num, device=x.device, dtype=x.dtype).index_add(0, index, torch.ones_like(index, dtype=x.dtype))
    mx = torch.full((B, num, d), float("-inf"), device=x.device, dtype=x.dtype)
    mx = mx.scatter_reduce(1, idx, x, reduce="amax", include_self=True)
    return s / cnt[None, :, None].clamp_min(1.0), mx


def default_anchors() -> torch.Tensor:
    """15 symmetric anchors in [-1, 1]^3: origin, 6 axis points, 8 octant points."""
    a = 0.5
    pts = [[0, 0, 0]]
    pts += [[s * a if i == ax else 0 for i in range(3)] for ax in range(3) for s in (-1, 1)]
    pts += [[sx * a, sy * a, sz * a] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return torch.tensor(pts, dtype=torch.float32)


class CrossScaleBlock(nn.Module):
    """H = S + MHA(S, U, U); S~ = H + FFN(LN(H))."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, s: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = s + self.attn(s, ctx, ctx, need_weights=False)[0]
        return h + self.ffn(self.ln(h))


class FarFieldBranch(nn.Module):
    def __init__(self, in_dim: int = 128, dim: int = 256, heads: int = 4, levels: int = 4,
                 anchors: torch.Tensor = None, sigma_kappa: float = 1.0):
        super().__init__()
        assert dim % heads == 0
        self.levels, self.sigma = levels, sigma_kappa
        self.inp = nn.Linear(in_dim, dim)
        # pooling projections W_Gamma: index L is points -> level-L cells, k < L is level k+1 -> k
        self.pool = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(levels + 1))
        self.cross = nn.ModuleList(CrossScaleBlock(dim, heads) for _ in range(levels + 1))
        self.register_buffer("anchors", default_anchors() if anchors is None else anchors)
        q = self.anchors.shape[0]
        # learnable kernel matrices W_q^{(k)}, stored as one (q*dim -> dim) map per level
        self.kernel = nn.ModuleList(nn.Linear(q * dim, dim, bias=False) for _ in range(levels + 1))

    def kernel_weights(self, local: torch.Tensor) -> torch.Tensor:
        """rho_q(p_i) = max(0, 1 - ||p~_i - zeta_q|| / sigma_kappa), shape (K, Q)."""
        return torch.relu(1.0 - torch.cdist(local, self.anchors) / self.sigma)

    def forward(self, j_near: torch.Tensor, tree: Octree) -> torch.Tensor:
        """j_near: (B, K, in_dim) near-field state -> J_far^{(L)} (B, K, dim)."""
        L = self.levels
        x = self.inp(j_near)
        S = [None] * (L + 1)
        mean, mx = scatter_mean_max(x, tree.point_cell[L], tree.num_cells[L])
        S[L] = self.pool[L](torch.cat([mean, mx], -1))                     # S^{(L)}
        for k in range(L - 1, 0, -1):
            mean, mx = scatter_mean_max(S[k + 1], tree.parent[k + 1], tree.num_cells[k])
            S[k] = self.pool[k](torch.cat([mean, mx], -1))

        St = [None] * (L + 1)
        St[1] = S[1]
        for k in range(2, L + 1):
            ctx = St[k - 1][:, tree.parent[k]]                             # U_k: one entry per level-k cell
            St[k] = self.cross[k](S[k], ctx)

        jf = St[1][:, tree.point_cell[1]]                                  # J_far^{(1)} at the points
        for k in range(2, L + 1):
            rho = self.kernel_weights(tree.local_coords[k])                # (K, Q)
            lifted = (rho[None, :, :, None] * jf[:, :, None, :]).flatten(2)  # (B, K, Q*dim)
            jf = St[k][:, tree.point_cell[k]] + self.kernel[k](lifted)
        return jf
