# U-PINet backbone

U-PINet maps sampled incident fields on a 3D perfectly electrically conducting (PEC)
surface to complex tangential surface currents. The backbone combines a near-field
graph encoder with geometric edge attention and cubic B-spline message kernels, and
a hierarchical branch with octree pooling, cross-scale attention, and kernel-point
fusion. The model adds the currents read out from the two branches.

## Installation

```bash
python -m pip install .
```

## Inputs and outputs

- Prepare your own facet centroids in metres and outward unit normals as arrays of
  shape `(K, 3)`. Call `prepare_target(points, normals, lambda_ref, kde_bandwidth)`
  with a fixed reference wavelength and a kernel-density bandwidth, both in metres.
  Use the same octree depth for the geometry and `UPINetConfig`.
- Supply the sampled `n × E_inc` in V/m as a complex PyTorch tensor of shape
  `(B, K, 3)`, in the same facet order. Each batch entry is an excitation of the
  same target. Put the model, geometry, and field tensor on the same device.
- `UPINet(config)(geometry, tangential_field)` returns a complex tensor of shape
  `(B, K, 3)` in A/m. With `return_branches=True`, it returns the total, near-field,
  and hierarchical current tensors in that order.

The default graph uses all points within half the fixed reference wavelength.
`max_neighbors` optionally limits each neighborhood to its closest points.
The model lifts the real and imaginary field channels into a hidden state, pools
near-field features into the finest octree cells, and projects both current readouts
onto the facet tangent planes.

## License

MIT. See [LICENSE](LICENSE).
