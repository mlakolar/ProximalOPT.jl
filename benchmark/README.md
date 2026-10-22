# Lasso benchmark

`lasso_bench.jl` times ProximalOPT's solvers on the lasso

    minimize  ‖y - Xβ‖² / (2n) + λ ‖β‖₁

against coordinate descent from
[CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl), with
glmnet (through GLMNet.jl) and Lasso.jl as outside references.

## Running

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/lasso_bench.jl --quick   # about 30 seconds
julia --project=benchmark benchmark/lasso_bench.jl           # 1.5 to 3.5 hours
julia --project=benchmark benchmark/lasso_summary.jl         # tables; --raw adds every run
```

The quick run checks that everything works on one small problem. The full run
writes `benchmark/results/single.csv` and `path.csv`, which
`lasso_summary.jl` turns into the tables below.

The environment needs Julia 1.11 or later, because it loads ProximalOPT from
this repository through `[sources]` in `Project.toml`. To benchmark a local
checkout of CoordinateDescent instead of the registered version, run
`Pkg.develop(path = "...")` in the environment. This writes the path into
`Project.toml`, so don't commit that change.

## What is measured

Every solver minimizes the same objective, without an intercept or
standardization. The designs have Gaussian columns scaled so that
`‖X[:, k]‖² = n`, `s` nonzero coefficients ±1 at random positions, and noise
with a signal-to-noise ratio of 4.

| design | n | p | s | correlation between columns |
|---|---:|---:|---:|---|
| A | 500 | 2,500 | 50 | none |
| B | 500 | 2,500 | 50 | AR(1), 0.8 between neighbors |
| C | 5,000 | 500 | 50 | AR(1), 0.5 between neighbors |
| D | 1,000 | 20,000 | 20 | none |
| E | 500 | 2,500 | 50 | 0.5 between every pair (a common factor) |

- **One λ** (0.1 λmax and 0.02 λmax), starting from β = 0.
  - Each solver runs over a sweep of its tolerance, from 1e-1 to 1e-16.
  - Each run records its time and its relative suboptimality `(F - F*) / F*`.
  - The tables give the time of the cheapest run that reaches the target.
  - `F*` comes from coordinate descent run to `optTol = 1e-14`, with a KKT residual of about 1e-14.
- **Defaults**: each package with its default options.
- **Path**: 50 λ values from λmax to λmax/100, each warm started from the previous solution. The tables give the time until every λ is within the target.
- **Timing**: each time is the minimum of at least three runs, with one BLAS thread.

The solvers:

| row | what runs |
|---|---|
| CD random, CD cyclic | `coordinateDescent!` with `randomize = true` (CoordinateDescent's default) or `false` |
| CD path from λmax | `coordinateDescent!` with `warmStart = false`, which solves along a path of 50 penalties |
| CD X'X/n | `coordinateDescent!` on `CDQuadraticLoss(X'X/n, -X'y/n)` |
| CD LassoPath | `CoordinateDescent.LassoPath` |
| ProxGrad, AccProxGrad, ActiveAccProxGrad | ProximalOPT's solvers on `LeastSquaresLoss` |
| AccProxGrad X'X/n, ActiveAccProxGrad X'X/n | the same on `QuadraticFunction(X'X/n, -X'y/n)` |
| glmnet, Lasso.jl | `GLMNet.glmnet` and `Lasso.fit(LassoPath, ...)` |

## Results

These results are from 22 October 2026, on an Apple M3 Pro with Julia 1.13.2 and one BLAS thread. Package versions:

- ProximalOPT at `07a0c3e`
- ProximalBase 0.4.0
- CoordinateDescent 0.5.1
- GLMNet.jl 0.7.4
- Lasso.jl 0.7.5

### One λ: milliseconds to a relative suboptimality of 1e-6

| solver | A, 0.1 | A, 0.02 | B, 0.1 | B, 0.02 | C, 0.1 | C, 0.02 | D, 0.1 | D, 0.02 | E, 0.1 | E, 0.02 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD cyclic | **3.4** | **144** | **3.7** | **65.8** | 1.3 | 6.6 | **17.1** | **277** | 2.0 | 209 |
| CD random | 6.8 | 172 | 4.9 | 147 | 1.5 | 7.0 | 53.9 | 425 | **1.1** | **12.1** |
| CD path from λmax | 20.4 | 311 | 21.8 | 427 | 26.5 | 48.4 | 154 | 688 | 10.3 | 38.1 |
| CD X'X/n | 17.4 | 713 | 19.0 | 307 | **0.1** | **0.4** | – | – | 4.2 | 38.0 |
| ProxGrad | 44.7 | 849 | 114 | 871 | 16.6 | 30.3 | 478 | 8359 | 122 | 4932 |
| AccProxGrad | 70.8 | 272 | 114 | 340 | 27.2 | 32.3 | 1296 | 5275 | 395 | 1915 |
| ActiveAccProxGrad | 30.2 | 262 | 56.9 | 200 | 14.0 | 39.7 | 434 | 3479 | 399 | 2761 |
| AccProxGrad X'X/n | 250 | 984 | 408 | 1251 | 1.9 | 2.4 | – | – | 1367 | ✗ |
| ActiveAccProxGrad X'X/n | 60.5 | 597 | 117 | 712 | 0.5 | 2.0 | – | – | 1372 | 5621 |
| glmnet | 37.3 | 1797 | 45.7 | 1529 | 16.7 | 50.0 | 145 | 5529 | 22.4 | 4053 |
| Lasso.jl | 143 | 626 | 133 | 596 | 42.0 | 80.9 | 20.5 | 565 | 15.1 | 181 |
| nonzeros in the solution | 278 | 452 | 221 | 424 | 54 | 227 | 333 | 865 | 40 | 302 |

Column headings give the design and λ/λmax. The X'X/n rows do not include
forming X'X/n, which takes 77 ms for A, 106 ms for B, 25 ms for C and 77 ms
for E; for D it would need 3.2 GB of memory. ✗ means the target was not
reached (here the best was 2e-6). `lasso_summary.jl` also prints the table for
a relative suboptimality of 1e-10.

### Path of 50 λ values: seconds until every λ is within the target

| solver | A, 1e-6 | B, 1e-6 | C, 1e-6 | D, 1e-6 | E, 1e-6 | A, 1e-10 | B, 1e-10 | C, 1e-10 | D, 1e-10 | E, 1e-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD LassoPath | 0.64 | 0.77 | 0.07 | **1.2** | **0.08** | 3.4 | 2.9 | 0.14 | 5.3 | **0.28** |
| CD X'X/n | 3.4 | 1.7 | **0.03** | – | 0.29 | 15.2 | 9.8 | **0.03** | – | 0.82 |
| ProxGrad | 9.8 | 13.8 | 0.53 | ✗ | ✗ | ✗ | ✗ | 1.1 | ✗ | ✗ |
| AccProxGrad | 1.3 | 1.7 | 0.64 | 17.2 | 8.1 | 4.8 | 6.9 | 1.2 | ✗ | ✗ |
| ActiveAccProxGrad | 0.42 | 0.36 | 0.26 | 1.3 | 0.98 | **1.2** | **1.6** | 0.54 | **4.6** | ✗ |
| ActiveAccProxGrad X'X/n | **0.36** | **0.33** | **0.03** | – | 0.49 | ✗ | ✗ | 0.04 | – | ✗ |
| glmnet | 2.5 | 2.0 | 0.38 | 3.2 | 2.1 | ✗ | ✗ | 0.63 | ✗ | ✗ |
| Lasso.jl | 0.56 | 0.71 | 0.08 | 1.4 | 0.13 | ✗ | 2.3 | 0.08 | 6.3 | 0.30 |

Here the X'X/n rows include forming X'X/n. ✗ means the target was not reached
within the sweep: tolerances down to 1e-14, or 90 seconds per solver.

### Default settings at λ = 0.02 λmax

| solver | A: relative gap | A: support errors | E: relative gap | E: support errors |
|---|---:|---:|---:|---:|
| CD `lasso()` | 4.9e-10 | 0 | 1.2e-11 | 0 |
| ProxGrad | 2.0e-3 | 92 | 0.35 | 934 |
| AccProxGrad | 9.4e-5 | 24 | 0.097 | 709 |
| ActiveAccProxGrad | 2.4e-5 | 17 | 3.0e-4 | 21 |
| glmnet | 9.0e-4 | 66 | 3.6e-4 | 24 |
| Lasso.jl | 3.2e-5 | 16 | 1.4e-6 | 1 |

A support error is a coefficient that is zero in one solution and nonzero in
the other. The solution has 452 nonzeros for A and 302 for E.
`lasso_summary.jl` prints all the default-settings tables.

## What the numbers say

- **For one λ, coordinate descent is fastest.**
  - At a relative gap of 1e-6 it is 1.8–25× faster than the best ProximalOPT solver on designs A–D.
  - On the equicorrelated design E it is 110–160× faster. There the common factor gives X'X/n one very large eigenvalue, which forces ProximalOPT's gradient methods to take small steps.
- **For a path, `ActiveAccProxGradDescent` is fastest when p > n and the columns are not strongly correlated.**
  - On A and B it beats `LassoPath` by 1.5–2.3× at a gap of 1e-6, and by 1.8–2.8× at 1e-10.
  - On D (p = 20,000) the two are about even: `LassoPath` is slightly faster at 1e-6 (1.2 s against 1.3 s), `ActiveAccProxGrad` at 1e-10 (4.6 s against 5.3 s). The strong-rule screening of CoordinateDescent 0.5 made `LassoPath` 2.2× faster here than CoordinateDescent 0.4.1.
  - On E it is the other way round: `LassoPath` is 12× faster, and `ActiveAccProxGrad` does not reach 1e-10.
- **When n ≫ p (design C), solve on X'X/n.** `CD X'X/n` and `ActiveAccProxGrad X'X/n` compute the whole path in 0.03 s, including forming X'X/n.
- **Within ProximalOPT, use `ActiveAccProxGradDescent`.**
  - `AccProxGrad` without the active set is up to 13× slower on paths: 17.2 s against 1.3 s for D.
  - `ProxGrad` is competitive only at low accuracy on easy problems.
  - The X'X/n variants pay off only when n ≫ p.
- **Coordinate order matters for coordinate descent.**
  - Cyclic order beats random order in all 8 cases on A–D, by up to 3.2×.
  - On E cyclic order is 17× slower (209 against 12.1 ms), which is why random order is CoordinateDescent's default.
- **ProximalOPT's defaults stop early on harder problems.** The defaults are `maxiter = 200` and `ftol = 1e-6`.
  - At 0.02 λmax on A–D its solvers stop with relative gaps of 1e-5 to 4e-3 and 16 to 260 support errors.
  - On E, `ProxGrad` and `AccProxGrad` stop with gaps of 0.35 and 0.1 and most of the support wrong.
  - For accurate solutions, set `ftol`, `xtol` and `gradtol` to about 1e-10 and raise `maxiter`.
  - CoordinateDescent's defaults reach gaps of 5e-10 or smaller, with no support errors, in every case.
- **glmnet's tolerance is on a different scale.** Its `tol` bounds squared changes in the coefficients, so reaching a gap of 1e-6 at every λ of a path needs a `tol` of about 1e-11. That makes it slow in these tables. At its default `tol`, its paths take a fraction of these times but are only accurate to about 1e-3.
