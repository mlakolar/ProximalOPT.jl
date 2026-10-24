# QUIC and the Kronecker caches

This directory contains two separate measurements for
[issue #22](https://github.com/mlakolar/ProximalSolvers.jl/issues/22).
`quic_bench.jl` compares graphical-lasso solvers. `kron_cache_bench.jl`
compares the two existing coordinate-descent caches on fixed symmetric
Kronecker quadratics. Run either script inside a Slurm allocation on
Discovery, with one Julia and BLAS thread.

## Existing graphical-lasso evidence

[PR #43](https://github.com/mlakolar/ProximalSolvers.jl/pull/43) records
Discovery job **13351907**, run `20261023T2358Z-ps-gap-quic`, using Julia
1.13.2 and an AMD EPYC 7513. Its tested source was `4579b53` on branch
`quic`. It compared proximal Newton, ADMM and GraphicalLassoQUIC.jl 0.1 at
five `(p, λ)` pairs: `(100, 0.2)`, `(100, 0.08)`, `(300, 0.2)`,
`(300, 0.08)` and `(600, 0.15)`. Each row records elapsed time and the
solution's relative duality gap, rather than assuming the solvers' tolerance
parameters give the same accuracy.

At a gap near `1e-4`, the recorded proximal-Newton times were
`0.005 / 0.021 / 0.050 / 0.277 / 0.269` seconds; the corresponding ADMM
times were `0.041 / 0.042 / 0.382 / 0.602 / 1.875` seconds. Proximal Newton
run to `1e-8` also beat GraphicalLassoQUIC at comparable measured gaps on
these cases. These are historical measurements from that run, not new
measurements of a changed checkout. They do not establish performance at
`p = 2000` or against R's QUIC or glasso, which were not in that comparison.

The run record labels the overall job `completed_with_failures`: an
unrelated A/B stage encountered a shared compiled-cache race and a summary
stage then failed. The graphical-lasso benchmark itself finished, and its
`quic_bench.txt` and `results/quic_bench.csv` were retained. A scheduler failure
for the combined job must not be described as a clean exit of every stage.

## Comparing `M = AΘB` with `U = ΘB`

`kron_cache_bench.jl` measures the cache choice before changing the public
`cdmodel` default. Both models solve exactly the same problem,

```
minimize  ⟨Θ, AΘB⟩ / 2 + ⟨C, Θ⟩ + g(Θ),  Θ symmetric.
```

`KronModel` computes a coordinate derivative in constant time and updates its
`M = AΘB` cache in quadratic time when the coordinate moves. `KronUModel`
uses linear time for both operations. A full pass through coordinates that
stay at zero can favor `KronModel`; active passes with many moves can favor
`KronUModel`. The benchmark records adverse cases as well as improvements.

The default grid is `p = 30, 100, 300`, with four interleaved repetitions.
The order of the two implementations alternates across repetitions. Each
receives the same initial point and a fresh identical RNG stream. Compilation
uses a separate small problem before timing. Model/iterate construction time
is recorded separately and included in the reported total.

The cases are:

- **D-trace**, the objective used by CovarianceSelection's direct difference
  estimator: `A = Sx`, `B = Sy`, `C = Sx − Sy`, and an entrywise ℓ1 penalty
  at `0.5` and `0.2` times `max(abs, Sx − Sy)`.
- **Reduced symmetric inverse-Hessian subproblems**, as represented in
  `benchmark/engine/new/run.jl`: `C = −(E₂₁ + E₁₂)/√2`, with the target
  coordinate unpenalized, fixed variance weights, and off-diagonal weights
  divided by `√2` to transform from `svec` coordinates. Two fixed penalty
  levels, `1` and `3`, probe different support sizes. The weights are the
  existing benchmark's lower-triangle gradient standard errors at the
  asymmetric starting point `E₂₁`. This is a cache comparison on fixed
  convex subproblems. It does not run DiffPrecTest's full reweighting or
  certify its statistical inference procedure.

`KronUModel` currently accepts a `SymmetricSparseIterate` only. Consequently,
the asymmetric inverse-Hessian problem on unrestricted `vec(Θ)` is outside
this comparison; this script does not claim an unsupported cache can solve it.

Every repetition records time, passes, convergence, the independently
evaluated objective, the package's scaled KKT residual and a direct entrywise
stationarity residual. Paired rows also report coefficient and objective
disagreements. The summary contains the best and median time ratios `U/M`;
ratios above one mean the U cache is slower. A validation failure preserves
the CSVs and exits with an error. The checks concern the stated subproblems,
not a proof that either cache is always faster.

### Running on Discovery

Use the existing `benchmark/quic` environment, with ProximalSolvers bound to
the checkout being measured and ProximalBase pinned in its manifest. Package
setup and all Julia execution must happen inside the Slurm allocation.

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
# Set a private depot layer for this run, ahead of the existing package cache.
export JULIA_DEPOT_PATH="$PWD/depot:/home1/mkolar/juliaPkg/depot:"
srun --exclusive --exact -N1 -n1 -c1 \
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia \
  --startup-file=no --project=/absolute/path/to/env-quic \
  /absolute/path/to/source/benchmark/quic/kron_cache_bench.jl \
  --sizes 30,100,300 --rounds 4 --out "$PWD/results/kron_cache.csv"
```

The environment already needs only `LinearAlgebra`, `Random`, `Printf`,
ProximalBase and ProximalSolvers for this cache script. It does not load
GraphicalLassoQUIC or R. A short check can use `--sizes 30 --rounds 2`.
Retain the raw and summary CSVs, the Slurm output, the tested source revisions
and a compact run recipe. New cache timings are **pending** until the script
has run on Discovery and its diagnostics have been inspected.
