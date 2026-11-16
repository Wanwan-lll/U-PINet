"""U-PINet backbone mapping sampled incident fields to tangential facet currents."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .far_field import FarFieldBranch
from .geometry import TargetGeometry
from .near_field import NearFieldEncoder


ETA0 = 4e-7 * pi * 299_792_458.0  # Free-space wave impedance in ohms.


@dataclass
class UPINetConfig:
    near_hidden: int = 128
    iterations: int = 3                 # T_iter
    kan_widths: Sequence[int] = (64, 64)
    grid_size: int = 5
    attn_hidden: int = 64
    far_dim: int = 256
    heads: int = 4
    octree_levels: int = 4              # L
    sigma_kappa: float = 1.0
    edge_chunk: int = 0                 # Maximum edges per message block. Zero uses all edges.


class UPINet(nn.Module):
    def __init__(self, cfg: Optional[UPINetConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = UPINetConfig()
        self.cfg = cfg
        self.near = NearFieldEncoder(6, cfg.near_hidden, cfg.iterations, cfg.kan_widths,
                                     cfg.grid_size, cfg.attn_hidden, cfg.edge_chunk)
        self.near_head = nn.Linear(cfg.near_hidden, 6)
        self.far = FarFieldBranch(cfg.near_hidden, cfg.far_dim, cfg.heads, cfg.octree_levels,
                                  sigma_kappa=cfg.sigma_kappa)
        self.far_head = nn.Linear(cfg.far_dim, 6)

    @staticmethod
    def _to_current(channels: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        current = torch.complex(channels[..., :3], channels[..., 3:])
        n = normals.to(current.dtype)[None]
        return (current - (current * n).sum(-1, keepdim=True) * n) / ETA0

    def forward(self, geo: TargetGeometry, tangential_field: torch.Tensor,
                return_branches: bool = False):
        """Map caller-supplied n × E_inc (B, K, 3) in V/m to complex currents in A/m."""
        if (tangential_field.ndim != 3 or tangential_field.shape[1:] != geo.points.shape
                or not tangential_field.is_complex()):
            raise ValueError("tangential_field must be a complex tensor of shape (B, K, 3)")
        if geo.octree.levels != self.cfg.octree_levels:
            raise ValueError("geometry octree depth must match UPINetConfig.octree_levels")
        x0 = torch.cat((tangential_field.real, tangential_field.imag), dim=-1).to(geo.normals.dtype)
        s_near = self.near(x0, geo.graph)
        s_far = self.far(s_near, geo.octree)
        j_near = self._to_current(self.near_head(s_near), geo.normals)
        j_far = self._to_current(self.far_head(s_far), geo.normals)
        j = j_near + j_far
        return (j, j_near, j_far) if return_branches else j
