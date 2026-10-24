# Coordinate descent: the new code against the code it replaces

ProximalSolvers (called ProximalOPT before its first release) took over
coordinate descent, penalty paths, reweighting and refits from
CoordinateDescent.jl, and the losses that CovSel.jl and DiffPrecTest.jl wrote
for it. This benchmark runs both on the same problems and compares them
case by case: the time, the passes, and the solutions.

- `problems.jl`: the problems, from standard libraries only, so that both
  environments draw the same data.
- `old-cd/`: CoordinateDescent 0.6.0, its last release, which uses momentum by
  default with the random order, on ProximalBase 0.4 (`run.jl`): the cases of
  CoordinateDescent's own estimators.
- `old/`: CovSel, DiffPrecTest and ProximalOPT 0.1 on CoordinateDescent 0.5.4
  and ProximalBase 0.4, pinned to their commits of 23 October 2026 (`run.jl`).
  At those commits CovSel and DiffPrecTest allow CoordinateDescent 0.4 and 0.5
  only.
- `new/`: ProximalSolvers from this repository on ProximalBase 0.5 (`run.jl`).
- `compare.jl`: the comparison; `ab.sh`: interleaved runs and the comparison.

## Running

From the root of the repository:

```
julia --project=benchmark/engine/old-cd -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/engine/old -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/engine/new -e 'using Pkg; Pkg.instantiate()'
benchmark/engine/ab.sh 2                      # two rounds of each, about 12 minutes
benchmark/engine/ab.sh 3 'dtrace|diffprec'    # only the cases whose key matches
```

Each round runs the old code, then the new code, in fresh processes, and writes
`results/old-cd_<stamp>.csv`, `results/old_<stamp>.csv` and
`results/new_<stamp>.csv` with the solutions in `.jls` files; the runs of
`old-cd/` and `old/` together make the old side. `compare.jl` takes the fastest run of each case on each side, since
load on the machine only ever adds time; on a machine shared with other work,
runs that are not interleaved can differ by 30% or more. One run of a side alone:

```
julia --project=benchmark/engine/new benchmark/engine/new/run.jl --only 'lasso \| A'
julia benchmark/engine/compare.jl results/old_<stamp>.csv results/new_<stamp>.csv
```

## What is compared

Each case solves one problem with one solver setting, under the same key in
both runs, from the same starting point and, for a random coordinate order, the
same seed. The new code draws its random orders with the same calls as
CoordinateDescent, and extrapolates with the same momentum, so where the two
take the same steps they visit the coordinates in the same order and stop after
the same pass.

| group | cases |
|---|---|
| lasso | designs A–E of `benchmark/lasso_bench.jl` at 0.1 and 0.02 λmax: cyclic and random order, the path of 50 levels from λmax (`warmStart = false` / `continuation = 50`), the quadratic on X'X/n, an intercept with standardized columns |
| path | 50 levels from λmax to λmax/100 with the strong rule (`LassoPath` / `solve_path`) |
| group | the group lasso by block coordinate descent, groups of 5 |
| sqrtlasso, scaledlasso, feasiblelasso | `sqrtLasso`, `scaledLasso`, `feasibleLasso!` / `SqrtLeastSquaresLoss`, `reweight!` |
| dtrace | CovSel's direct difference estimator at p = 20, 50, 100 (`CDDirectDifferenceLoss` / `KronQuadratic` on a `SymmetricSparseIterate`), and a path of 20 levels |
| refit | CovSel's `differencePrecisionRefit`, by coordinate descent with the entries off the support penalized by 1e10 / `refit!`, exact |
| diffprectest | the first stages of DiffPrecTest's inverse Hessians at fixed weights (`CDInverseKroneckerLoss`, `CDInverseReducedSymKroneckerLoss` / `KronQuadratic` on vec Θ and on a symmetric Θ) |
| proxgrad, admm | the gradient solvers and ADMM of ProximalOPT 0.1 and of this package, before and after the port to ProximalBase 0.5. The new side also times `AccProxGradDescent` and `ActiveAccProxGradDescent` on designs A–E at 0.1 and 0.02 λmax, 500 iterations of `ProximalGradientDescent`, the logistic lasso on design A, and warm-started paths of 20 levels on designs A and D |

| iht | CovSel's `differencePrecisionIHT` (fixed step) / `ProximalGradientDescent` with `IndBallL0` on a `SymmetricSparseIterate` (backtracking) |

The old run also times cases with no counterpart yet, such as DiffPrecTest's
whole estimators.

`new/` takes ProximalBase 0.5 from the registry and ProximalSolvers from this
checkout.

## Results

23 October 2026 on USC CARC Discovery: one core of an AMD EPYC 7513, Julia
1.13.2, one BLAS thread, three interleaved rounds of each side (Slurm job
13326700). The new code is this package, then ProximalOPT, at `c247846`, on
ProximalBase 0.5.0.
The old side is CoordinateDescent 0.6.0 with its default momentum for
CoordinateDescent's own cases (`old-cd/`), and CovSel, DiffPrecTest and
ProximalOPT 0.1 on CoordinateDescent 0.5.4 for the others (`old/`). Ratios are
new over old times.

| group | cases | time ratio, geometric mean (range) | passes | largest difference of the solutions |
|---|---:|---|---|---|
| lasso | 34 | 0.91 (0.62–1.00) | the same, except one case (see below) | 8e-14; 3e-5 in that case |
| path | 5 | 0.94 (0.91–0.99) | the same | 5e-14 |
| group | 4 | 0.99 (0.97–1.01) | the same | 7e-15 |
| sqrtlasso | 4 | 0.92 (0.87–0.96) | the same | 3e-15 |
| scaledlasso, feasiblelasso | 4 | 0.97 (0.94–1.00) | the same | 2e-15 |
| dtrace | 16 | 0.95 (0.75–1.03) | within 15% | 2e-7 |
| diffprectest | 4 | 0.21 (0.10–0.35) | within 2 | 7e-8 |
| refit | 1 | 0.16 | | 1e-7 |
| iht | 1 | 0.22 | 33 against 453 | see below |
| proxgrad | 4 | 0.42 (0.14–0.81) | fewer | 5e-4, at a lower objective |
| admm | 4 | 0.98 (0.97–0.98) | the same | 1e-14 |

All 81 cases: 0.80. No case is more than 3% slower.

- The lasso, path, group, square-root, scaled and feasible lasso take the same
  steps as CoordinateDescent 0.6.0, momentum included: the same passes, and
  solutions that agree to rounding.
  - The exception is the path of 50 levels from λmax on design D. It takes 2601
    passes against 2559, in 0.97 of the time, and the objectives agree to 8e-12.
  - The new code scales the gradient of screening by `s = 1/n`, where
    CoordinateDescent divides by n. On this CPU, the rounding turns one
    borderline decision of the strong rule the other way. On the M3 Pro the
    passes are the same.
- The direct difference estimator takes a few passes more or less: CovSel's
  coordinate step stores a coordinate that ends at zero, which reorders the
  active coordinates after `dropzeros!`; the new step stores only nonzeros. Both
  stop at the tolerance, 1e-7 relative.
- In a first run on Discovery, the cyclic order on the direct difference
  estimator at p = 20 to 100 took 1.10–1.25 times as long. Each visit of a
  coordinate computed `tril_cartesian` twice, for the step and for the scale of
  the stopping rule. The scale is now computed only for a coordinate that moved,
  which most coordinates of a full pass do not.
- DiffPrecTest's losses indexed `Symmetric` matrices in their O(p²) cache
  updates; `KronQuadratic` keeps dense copies, and its symmetric form needs
  no √2 bookkeeping. At p = 50 the reduced symmetric loss goes from 13.8 to
  1.3 ms.
- `refit!` solves the D-trace loss on the support exactly, by the Hessian on
  the support, where `differencePrecisionRefit` ran coordinate descent with
  1e10 penalties off the support (and agrees with the exact refit to 1e-7).
- Projected gradient with backtracking stops at a lower objective than CovSel's
  IHT with its fixed step (−47.45 against −46.20; both are local solutions of a
  nonconvex problem with the same constraint, at most s nonzero entries in the
  lower triangle), in 33 instead of 453 iterations.
- The accelerated gradient solvers let their curvature estimate `L` decrease
  before each iteration (`α = 0.9`, as in TFOCS), where ProximalOPT 0.1 only let
  it grow: `AccProxGradDescent` takes 27 instead of 61 ms on A and 44 instead
  of 308 ms on E, at tolerance 1e-10, and stops at a lower objective. With
  `α = 1` they take the same steps as before.
- `ADMMSolver(; ρ = s², α = 1.5, adaptive = true, abstol = 1e-5/s,
  abstol_dual = 1e-5 s, reltol = 1e-3)`, with `s = tr(S)/p`, is CovSel's
  `covsel!`: the same iterations, and the same solution up to rounding. The
  default `ADMMSolver` takes the same iterations as before the port.

Beyond the old code, both sides new code:

- Momentum, measured on Discovery (job 13326719): coordinate descent on a
  logistic lasso is 1.4–5.6 times faster with it, and `ProxNewton`, whose inner
  solves are weighted least squares, 1.0–4.5 times. On `KronQuadratic` it is
  off by default. On vec Θ with the two nonzero entries of DiffPrecTest's
  inverse Hessians it was 1.08–1.17 times slower. With `momentum = true` the
  D-trace loss of a symmetric Θ is 0.78–0.93 of the time.
- `CDSolver(; order = :cyclic, anderson = 5)` against cyclic order alone at
  0.02 λmax, on the M3 Pro: 2.3× faster on A (224 → 96 ms), 2.2× on B, 1.6× on
  D and 2.6× on E (1468 → 562 ms), and 1.2× on the direct difference estimator
  at p = 50 and 100, at the same or a lower objective. It does not help a random
  order, which ignores it.

Earlier, before momentum moved here, the comparison ran on the M3 Pro against
CoordinateDescent 0.5.4: 0.83 over the same 81 cases, no case more than 4%
slower. The history of this file has its table.
