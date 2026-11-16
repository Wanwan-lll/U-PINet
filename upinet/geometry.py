"""Radius graph and octree built from caller-supplied surface geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class SurfaceGraph:
    src: torch.Tensor          # (E,) j
    dst: torch.Tensor          # (E,) i  (messages flow j -> i)
    dp: torch.Tensor           # (E, 3) relative position / lambda_ref
    dn: torch.Tensor           # (E, 3) normal difference
    inv_density: torch.Tensor  # (K,) d_j in [0, 1]
    num_nodes: int

    def to(self, device):
        return SurfaceGraph(self.src.to(device), self.dst.to(device), self.dp.to(device),
                            self.dn.to(device), self.inv_density.to(device), self.num_nodes)


def build_graph(points: np.ndarray, normals: np.ndarray, r_nf: float, lambda_ref: float,
                kde_bandwidth: float, max_neighbors: Optional[int] = None) -> SurfaceGraph:
    tree = cKDTree(points)
    if max_neighbors is None:
        nbr = tree.query_ball_point(points, r=r_nf)
        dst = np.concatenate([np.full(len(v), i) for i, v in enumerate(nbr)])
        src = np.concatenate([np.asarray(v, dtype=np.int64) for v in nbr])
    else:
        # nearest max_neighbors points inside r_nf (self included); keeps N(i) within radius r_nf
        k = min(max_neighbors, len(points))
        dist, idx = tree.query(points, k=k, distance_upper_bound=np.nextafter(r_nf, np.inf))
        ok = np.isfinite(dist.reshape(-1))
        dst = np.repeat(np.arange(len(points)), k)[ok]
        src = idx.reshape(-1)[ok]
    d = points[dst] - points[src]
    r2 = (d ** 2).sum(1)
    h = kde_bandwidth
    n_i = np.bincount(dst, minlength=len(points)).astype(float)
    dens = np.bincount(dst, weights=np.exp(-r2 / (2 * h * h)), minlength=len(points))
    dens = dens / (n_i * (2 * np.pi) ** 1.5 * h ** 3)
    inv = 1.0 / dens
    inv = inv / inv.max()
    f = lambda x: torch.as_tensor(x, dtype=torch.float32)
    return SurfaceGraph(torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long),
                        f(d / lambda_ref), f(normals[dst] - normals[src]), f(inv), len(points))


@dataclass
class Octree:
    point_cell: List[torch.Tensor]    # level k -> (K,) cell index of every point
    parent: List[torch.Tensor]        # level k -> (M_k,) parent cell index at level k-1 (k >= 2)
    local_coords: List[torch.Tensor]  # level k -> (K, 3) p~_i^{(k)} in [-1, 1]^3 w.r.t. level-(k-1) cell
    num_cells: List[int]              # M_k
    levels: int

    def to(self, device):
        mv = lambda xs: [None if x is None else x.to(device) for x in xs]
        return Octree(mv(self.point_cell), mv(self.parent), mv(self.local_coords), self.num_cells, self.levels)


def build_octree(points: np.ndarray, levels: int) -> Octree:
    """Reserve index 0 so that list indices match octree levels."""
    lo, hi = points.min(0), points.max(0)
    center = 0.5 * (lo + hi)
    size = 1.0001 * float((hi - lo).max())                             # full edge of the root cube
    if size == 0:
        size = 1.0  # A coincident point set fits in any cube centered on it.
    origin = center - 0.5 * size
    point_cell, parent, local, num_cells = [None], [None], [None], [0]
    centers, coords_by_level = [None], [None]
    for k in range(1, levels + 1):
        n = 2 ** k
        ijk = np.clip(np.floor((points - origin) / (size / n)).astype(np.int64), 0, n - 1)
        key = (ijk[:, 0] * n + ijk[:, 1]) * n + ijk[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        cell_ijk = np.stack([uniq // (n * n), (uniq // n) % n, uniq % n], 1)
        centers.append(origin + (cell_ijk + 0.5) * (size / n))
        coords_by_level.append((uniq, cell_ijk))
        point_cell.append(torch.as_tensor(inv.reshape(-1), dtype=torch.long))
        num_cells.append(len(uniq))
        if k >= 2:
            puniq, _ = coords_by_level[k - 1]
            pijk = cell_ijk // 2
            pkey = (pijk[:, 0] * (n // 2) + pijk[:, 1]) * (n // 2) + pijk[:, 2]
            parent.append(torch.as_tensor(np.searchsorted(puniq, pkey), dtype=torch.long))
            # local coordinates of each point inside its level-(k-1) cell
            pc = centers[k - 1][point_cell[k - 1].numpy()]
            half = 0.5 * size / (n // 2)
            local.append(torch.as_tensor((points - pc) / half, dtype=torch.float32))
        else:
            parent.append(None)
            local.append(None)
    return Octree(point_cell, parent, local, num_cells, levels)


@dataclass
class TargetGeometry:
    """Everything the network needs about one target."""
    points: torch.Tensor     # (K, 3) facet centroids [m]
    normals: torch.Tensor    # (K, 3)
    graph: SurfaceGraph
    octree: Octree
    lambda_ref: float

    def to(self, device):
        return TargetGeometry(self.points.to(device), self.normals.to(device),
                              self.graph.to(device), self.octree.to(device), self.lambda_ref)


def prepare_target(points: np.ndarray, normals: np.ndarray, lambda_ref: float,
                   kde_bandwidth: float, *, r_nf_wavelengths: float = 0.5,
                   octree_levels: int = 4, max_neighbors: Optional[int] = None) -> TargetGeometry:
    """Prepare a radius graph and octree from facet centroids and outward unit normals."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or normals.shape != points.shape:
        raise ValueError("points and normals must have matching nonempty shape (K, 3)")
    if not (np.isfinite(points).all() and np.isfinite(normals).all()):
        raise ValueError("points and normals must be finite")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5):
        raise ValueError("normals must be unit vectors")
    scales = [lambda_ref, kde_bandwidth, r_nf_wavelengths]
    if not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("lambda_ref, kde_bandwidth and r_nf_wavelengths must be finite and positive")
    if octree_levels < 1 or (max_neighbors is not None and max_neighbors < 1):
        raise ValueError("octree_levels and max_neighbors must be positive")
    graph = build_graph(points, normals, r_nf_wavelengths * lambda_ref,
                        lambda_ref, kde_bandwidth, max_neighbors)
    tree = build_octree(points, octree_levels)
    return TargetGeometry(torch.as_tensor(points, dtype=torch.float32),
                          torch.as_tensor(normals, dtype=torch.float32), graph, tree, lambda_ref)
