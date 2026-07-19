# Phase 4: Support-Aware Factorial Cokriging

## Mathematical model

For `p` colocated variables and `K` spatial structures, GeoSupportFM uses the
Linear Model of Coregionalization

```text
Gamma(h) = sum_k B_k g_k(h)
C(h)     = sum_k B_k rho_k(h)
```

where `g_k(h) = 1 - rho_k(h)`. Each `B_k` is constrained to be positive
semidefinite. This is required for the assembled multivariate covariance matrix
to remain permissible. `fit_lmc` enforces the constraint by optimizing lower
triangular factors `L_k` and calculating `B_k = L_k L_k.T`.

## Change of support

A support is stored as quadrature offsets `u_a` and normalized weights `w_a`.
The covariance between supports centered at `x` and `y` is approximated by

```text
C_VW(x, y) = sum_a sum_b w_a w_b C((x + u_a) - (y + u_b)).
```

This one operator covers point-to-point, point-to-block, and block-to-block
covariance. `regularize_lmc` calculates the block covariance profile, block
variogram, point and block sill matrices, variance ratios, and standard
deviation correction ratios. Increasing quadrature density improves numerical
integration and increases computational cost quadratically.

## Factorial cokriging

The data-to-data matrix contains the full LMC. To estimate factor `k`, the
data-to-target covariance contains only `B_k rho_k(h)`. With constant drift
matrix `F`, the system for a zero-mean factor is

```text
[C  F] [W ] = [C_k]
[F' 0] [mu]   [ 0 ]
```

The zero right-hand constraint removes variable-specific unknown constants.
For ordinary cokriging of the full field, the constraint becomes an identity
for the target variable. For centered data with a known mean,
`simple_cokriging` omits drift constraints.

The implementation uses observation-major ordering:

```text
[location_0 variable_0, location_0 variable_1, location_1 variable_0, ...]
```

## Workflow

1. Calculate direct and cross-semivariograms with
   `empirical_variogram_matrix`.
2. Define initial nested structures and fit permissible matrices with
   `fit_lmc`.
3. Represent observation and prediction supports with `SpatialSupport`.
4. Inspect support-induced variance reduction with `regularize_lmc`.
5. Estimate individual scales using `factorial_cokriging`.
6. Estimate the complete field with `ordinary_cokriging` or
   `simple_cokriging`.
7. Aggregate latent vectors with `aggregate_embeddings_to_blocks`. Enable
   `unit_normalize` when the embedding model requires unit-length vectors.

The executable example is `examples/phase4_factorial_cokriging.py`.

## Numerical choices

- Coordinates and support sizes must use the same projected distance units.
- Every coregionalization matrix is validated for symmetry and positive
  semidefiniteness.
- A small diagonal ridge stabilizes covariance solves. The returned condition
  number should be monitored when selecting neighborhoods and nested models.
- Factorial components are probabilistic filters at sampled locations. Their
  purpose is scale decomposition, so individual factors do not honor the
  composite observations exactly.
- Nugget variance under a discretized block is attenuated by the quadrature
  weights. Equal weights give an attenuation of `1 / n_discretization_points`.

## Sources in this repository

- Ma, Royer, Wang, Wang, and Zhang (2014), *Factorial kriging for multiscale
  modelling*.
- Silva, Grego, Manzione, and Oliveira, *Taking into account the change of
  support when fusing heterogeneous spatial data for spatial interpolation*.

