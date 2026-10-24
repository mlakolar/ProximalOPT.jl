# QUIC and the Kronecker caches

This directory contains three separate measurements for
[issue #22](https://github.com/mlakolar/ProximalSolvers.jl/issues/22).
`quic_bench.jl` compares graphical-lasso solvers. `kron_cache_bench.jl`
compares the two existing coordinate-descent caches on fixed symmetric
Kronecker quadratics. `cross_language.jl` and `cross_language.R` compare
warm-started graphical-lasso paths across five Julia and R solvers.
Run these scripts inside a Slurm allocation on
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

## Cross-language graphical-lasso paths

Discovery job **13377414**, run `20261024T1258Z-ps-issue22`, measured
`p = 100, 300, 600, 2000`, three decreasing penalty levels
`λ = 0.2, 0.12, 0.08`, and two interleaved repetitions. Its tested source
was `b075cdc4eba160dffb97dfd9a2c3318f5c7d4d73`, with ProximalBase
`a866768afcca653503af7b174d2fabc3b7c38466`. The solvers were native QUIC,
ADMM (the algorithm used by `covsel!`), GraphicalLassoQUIC.jl, R's QUIC and
R's glasso. The common objective was

```
tr(SΘ) − log det Θ + λ Σᵢ≠ⱼ |Θᵢⱼ|.
```

Each implementation received the same binary-exported sample covariance.
The diagonal was unpenalized, with no implicit likelihood rescaling.
Every path reused its previous primal solution. ADMM also reused its dual
variable and penalty parameter; the two reference QUIC interfaces received
the previous precision and its inverse. GraphicalLassoQUIC's required
inverse construction was included in its timed work. Solver order reversed
between repetitions. R startup, file I/O and Julia validation were outside
the solve times.

Julia checked every returned matrix's symmetry, positive definiteness,
objective and relative duality gap. Objective and gap evaluation used the
symmetric part; the symmetry test applied to the original output.
**118 of 120 results passed.** Both
R glasso results at `p = 2000, λ = 0.08` failed the fixed symmetry check:
relative asymmetry `1.309748671e-8` exceeded `1e-8`. Their evaluated relative
duality gaps were approximately `4.219e-10`. The small gaps do not override
the symmetry failure. These two results remain failed and must be excluded
from accuracy-matched timing claims. The benchmark exited with an error
after preserving every result; it was not a clean all-pass validation run.

The raw `validated.tsv` contains every time, attained gap and validation
decision, including the failures. A summary should report both all-returned
and valid-only median times, with the number of valid repetitions. Solver
tolerance names alone do not establish equal accuracy; the common target
was a relative gap of `1e-4`, and attained gaps were recorded separately.

Median solve times in seconds, from the two interleaved repetitions:

| p | λ | Native QUIC | ADMM | GraphicalLassoQUIC | R QUIC | R glasso |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 0.20 | 0.000303 | 0.008721 | 0.003019 | 0.0025 | 0.0020 |
| 100 | 0.12 | 0.001204 | 0.008582 | 0.006252 | 0.0020 | 0.0065 |
| 100 | 0.08 | 0.002388 | 0.008574 | 0.016459 | 0.0040 | 0.0125 |
| 300 | 0.20 | 0.001151 | 0.087520 | 0.026216 | 0.0150 | 0.0080 |
| 300 | 0.12 | 0.009456 | 0.088778 | 0.034169 | 0.0130 | 0.1430 |
| 300 | 0.08 | 0.011274 | 0.091342 | 0.085480 | 0.0210 | 0.1560 |
| 600 | 0.20 | 0.002481 | 0.491594 | 0.099909 | 0.0690 | 0.0330 |
| 600 | 0.12 | 0.050782 | 0.510850 | 0.166664 | 0.0655 | 0.8695 |
| 600 | 0.08 | 0.047998 | 0.515336 | 0.241428 | 0.0810 | 1.1890 |
| 2000 | 0.20 | 0.036533 | 12.865421 | 1.992394 | 1.4425 | 0.7070 |
| 2000 | 0.12 | 1.128101 | 13.358484 | 3.083762 | 1.5325 | 35.9685 |
| 2000 | 0.08 | 1.180265 | 13.493029 | 3.958057 | 1.7825 | 35.7235* |

`*` Both glasso repetitions failed the symmetry check described above; their
time is retained for completeness, not an accuracy-matched comparison.
Across the 12 accepted native/ADMM and native/GraphicalLassoQUIC pairs, the
ratios of median times were `0.00284–0.278` and `0.0183–0.366`, respectively.
Thus the native solver met the issue's two timing thresholds on this grid.
These are times to the common tolerance target, not to identical attained
gaps: the reference QUIC implementations often solved more accurately.

### Recreating the cross-language run

Use the run's R library containing `QUIC` and `glasso`; the R QUIC package
is named `QUIC`, not `cQUIC`. This run used R 4.4.3.
Its QUIC source came from the
[CRAN mirror at `1235758`](https://github.com/cran/QUIC/tree/1235758fe364d275565f11d2ca96276d7eadc6ba).
The reference parameter and warm-start conventions are documented in the
[QUIC source](https://github.com/cran/QUIC/blob/1235758fe364d275565f11d2ca96276d7eadc6ba/R/QUIC.R)
and [glasso manual](https://cran.r-project.org/web/packages/glasso/glasso.pdf).
Keep package installation and all execution inside the Slurm allocation.

```sh
module load r/4.4.3
export R_RUNTIME_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export R_LIBS_USER="/absolute/path/to/run/Rlib"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
R_BINARY=$(command -v Rscript)
srun --exclusive --exact -N1 -n1 -c1 env -u LD_LIBRARY_PATH \
  /home1/mkolar/juliaPkg/julia-1.13.2/bin/julia \
  --startup-file=no --project=/absolute/path/to/env-quic \
  /absolute/path/to/source/benchmark/quic/cross_language.jl \
  --root "$PWD/cross-results" --sizes 100,300,600,2000 \
  --lambda 0.2,0.12,0.08 --rounds 2 --rscript "$R_BINARY"
```

The Julia process excludes R's module library path; each R child receives
it through `R_RUNTIME_LIBRARY_PATH`. This avoids an observed library-version
conflict between R's modules and Julia's bundled tools. The script records
missing packages as unavailable and preserves failed results. Use
`--validate-only` with the same input options to inspect saved matrices
without refitting. ADMM has a 120-second soft limit per level; Newton and
reference iterations are capped, and each R child path has a process limit.

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

### Cache results

Discovery job **13377133**, tested source
`f99d014494b8106f1c2a02742dc94226cbbf000e`, completed all 12 cases with
validation passed. The largest relative coefficient discrepancy was
`8.8e-14`; the largest scaled KKT residual was `4.8e-9`. Median total-time
ratios `U/M` were:

| p | D-trace, 0.5 λmax | D-trace, 0.2 λmax | Inverse Hessian, level 1 | Inverse Hessian, level 3 |
|---|---:|---:|---:|---:|
| 30 | 0.289 | 0.350 | 0.346 | 0.512 |
| 100 | 0.226 | 0.115 | 0.111 | 0.153 |
| 300 | 0.145 | 0.0505 | 0.0466 | 0.0686 |

The U cache was faster on these fixed symmetric subproblems. The raw
repetitions and summary are retained as `kron_cache.csv` and
`kron_cache_summary.csv` in the run's results. The evidence does not cover
unrestricted asymmetric iterates or all possible active-set patterns.

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
and a compact run recipe.
