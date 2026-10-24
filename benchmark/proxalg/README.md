# ProximalAlgorithms.jl against ProximalSolvers

Issues #23 (a primal–dual solver), #24 (nonconvex gradient methods) and #25
(PANOC, ZeroFPR, PANOCplus) say to measure
[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl)
before writing any of these methods here. This folder does that. It wraps
ProximalBase functions for ProximalAlgorithms 0.7 (on ProximalCore 0.1) in
`adapter.jl`, and runs both packages' solvers on the same objects from the same
starting points. It is a separate environment; nothing here is a dependency of
ProximalSolvers.

## Running

```
julia --project=benchmark/proxalg -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/proxalg benchmark/proxalg/run.jl --quick   # a few minutes
julia --project=benchmark/proxalg benchmark/proxalg/run.jl           # about 30 minutes
julia --project=benchmark/proxalg benchmark/proxalg/summary.jl       # the tables below
```

Results go to `benchmark/proxalg/results/`, or to the folder in
`PROXALG_OUT`. `PROXALG_ONLY="IHT,MCP"` runs only those problem families, and
adding `probe` runs the probe. The environment needs Julia 1.11 or later,
because it loads ProximalSolvers from this repository through `[sources]`.
When the checked-out ProximalSolvers has the methods of #24
(`MonotoneAccProxGradDescent`, `ProximalGradientDescent`'s `normalized` and
`debias`), they join the runs.

## What is measured

| problem | objective | convex | sizes |
|---|---|---|---|
| graphical lasso | `LogDetLoss(S)` + `λ‖X‖₁` off the diagonal, from `X = I` | yes, gradient only locally Lipschitz, `f = Inf` off the cone | p = 50, 200 |
| IHT, least squares | `‖y − Xβ‖²/(2n)` with `‖β‖₀ ≤ s` | no | n = 500, p = 2000, s = 20 |
| IHT, D-trace | `tr(ΣxΔΣyΔ)/2 − tr(Δ(Σy − Σx))`, at most `s` off-diagonal entries of a symmetric `Δ` | no | p = 30, s = 10 |
| MCP | `‖y − Xβ‖²/(2n) + Σⱼ MCP(βⱼ; λ, 3)` | no | n = 500, p = 2000 |
| quantile regression | `(1/n) Σᵢ ρ_τ(yᵢ − xᵢ'β) + λ‖β‖₁`, t₃ noise | yes, nonsmooth | n = 500, p = 100; n = 2000, p = 200 |
| TV denoising | `‖y − β‖²/2 + λ‖Dβ‖₁`, `D` the differences | yes | n = 1000, 10000 |

The solvers:
- **ProximalSolvers:** `ProximalGradientDescent`, `AccProxGradDescent`, `ADMMSolver(; adaptive = true)` for the graphical lasso, `CDSolver` for MCP, and `AccProxGradDescent` on the dual `min ‖y − D'u‖²/2` subject to `‖u‖∞ ≤ λ` for TV.
- **ProximalAlgorithms, with their default (adaptive) steps:**
  - `ForwardBackward`, `FastForwardBackward`, `PANOC`, `ZeroFPR` and `PANOCplus`;
  - `LiLin`, with the fixed step `1/L` it requires;
  - `ChambollePock` and `VuCondat`, with steps from `‖D‖` or `‖X‖`.
- **JuMP + HiGHS (simplex):** the linear program for quantile regression.

**Accuracy:**
- For a convex problem, each solver runs over a sweep of its own tolerance (1e-1 to 1e-12), and each run records its time, iterations and objective. The tables give the time of the cheapest run that reaches the relative suboptimality `(F − F*)/|F*|`. `F*` comes from a tight reference run (ADMM, the LP, or the dual solver), or from any run that does better.
- For a nonconvex problem, `F_best` is the lowest objective any run reached, and the tables give how far each solver's best run stayed from it.
- Times are the minimum of three runs (one run when slower than 30 s).
- Every run is capped at 60 s.

**Bytes per iteration** come from runs of exactly 20 and 220 iterations. **Time per iteration** comes from the sweep run with the most iterations.

**The probe** starts the graphical lasso with a first step that leaves the
positive definite cone: ProximalAlgorithms' solvers with an initial step
`γ₀ = 1` or `10`, which they backtrack from, and ProximalSolvers' gradient
solvers, whose first trial step is 1.

## Results

Discovery (USC CARC), AMD EPYC 7513, one BLAS thread, Julia 1.13.2;
ProximalSolvers at `d16afd9` (master then), ProximalBase 0.5.2,
ProximalAlgorithms 0.7.0, ProximalCore 0.1.2; run
`20261023T2359Z-ps-proxalg-bench`.

### Graphical lasso

| p | solver | package | time to 1e-4 | time to 1e-6 | time to 1e-8 | bytes/iteration | time/iteration | note |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 50 | ProxGrad | ProximalSolvers | 9.3 ms | 16.9 ms | 25 ms | 0 | 0.039 ms | |
| 50 | AccProxGrad | ProximalSolvers | 1.88 ms | 3.43 ms | 4.94 ms | 0 | 0.04 ms | |
| 50 | ADMM | ProximalSolvers | 4.86 ms | 8.28 ms | 12.5 ms | 0 | 0.2 ms | |
| 50 | ForwardBackward, FastForwardBackward | ProximalAlgorithms | — | — | — | 20–40 KB | 0.03–0.05 ms | ends at F = NaN |
| 50 | PANOC, ZeroFPR, PANOCplus | ProximalAlgorithms | — | — | — | 0.42–0.85 MB | 0.6–1.2 ms | ends at F = NaN |
| 200 | ProxGrad | ProximalSolvers | 151 ms | 261 ms | 344 ms | 0 | 0.83 ms | |
| 200 | AccProxGrad | ProximalSolvers | 33.5 ms | 55.7 ms | 81.1 ms | 0 | 0.86 ms | |
| 200 | ADMM | ProximalSolvers | 136 ms | 222 ms | 320 ms | 0 | 3.27 ms | |
| 200 | ForwardBackward, FastForwardBackward | ProximalAlgorithms | — | — | — | 0.32–0.64 MB | 0.6–1.2 ms | ends at F = NaN |
| 200 | PANOC, ZeroFPR, PANOCplus | ProximalAlgorithms | — | — | — | 6.7–13.4 MB | 14–26 ms | ends at F = NaN |

The probe gives the same picture. From a first step that leaves the cone:
- ProximalSolvers' `ProxGrad` and `AccProxGrad` reach `F − F* ≤ 7e-7` in 3.8–295 ms (p = 50 and 200);
- every ProximalAlgorithms solver, from `γ₀ = 1` or `10`, ends at `F = NaN` after 100,000 iterations or 60 s.

### Iterative hard thresholding

Least squares, n = 500, p = 2000, s = 20:

| solver | package | best F − F_best (rel.) | time to F_best + 1e-6 | iterations | bytes/iteration | note |
|---|---|---:|---:|---:|---:|---|
| ProxGrad | ProximalSolvers | 0 | 3.18 ms | 10 | 0 | |
| AccProxGrad | ProximalSolvers | 0 | 3.33 ms | 15 | 0 | |
| ForwardBackward | ProximalAlgorithms | 0 | 4.73 ms | 32 | 16 KB | |
| FastForwardBackward | ProximalAlgorithms | 0 | 9.36 ms | 42 | 32 KB | |
| PANOC | ProximalAlgorithms | 0 | 4.82 ms | 13 | 32 KB | |
| ZeroFPR | ProximalAlgorithms | 0 | 2.70 ms | 8 | 32 KB | |
| LiLin (fixed step 1/L) | ProximalAlgorithms | 0 | 29.5 ms | 180 | 32 KB | |

D-trace, p = 30, s = 10:

| solver | package | best F − F_best (rel.) | time to F_best + 1e-6 | iterations | bytes/iteration | note |
|---|---|---:|---:|---:|---:|---|
| ProxGrad | ProximalSolvers | 0 | 0.36 ms | 33 | 0 | |
| AccProxGrad | ProximalSolvers | 0 | 0.45 ms | 32 | 0 | |
| ForwardBackward, FastForwardBackward, PANOC, ZeroFPR | ProximalAlgorithms | 7.1e-2 | — | 12–54 | 7–15 KB | a worse local minimum |
| LiLin | ProximalAlgorithms | 1.2e-1 | — | 166 | — | throws `UndefVarError: x` |

### Least squares with MCP

n = 500, p = 2000, λ = 0.05 λmax, a = 3:

| solver | package | best F − F_best (rel.) | time to its best | iterations | bytes/iteration |
|---|---|---:|---:|---:|---:|
| CD | ProximalSolvers | 3.2e-4 | 17.3 ms | 2097 passes | 0 |
| ProxGrad | ProximalSolvers | 2.5e-2 | 229 ms | 655 | 0 |
| AccProxGrad | ProximalSolvers | 5.2e-3 | 102 ms | 291 | 0 |
| ForwardBackward | ProximalAlgorithms | 3.0e-2 | 3.93 s | 23192 | 16 KB |
| FastForwardBackward | ProximalAlgorithms | 2.3e-3 | 581 ms | 2791 | 32 KB |
| PANOC | ProximalAlgorithms | 0 | 384 ms | 726 | 34 KB |
| ZeroFPR | ProximalAlgorithms | 4.9e-3 | 89.9 ms | 146 | 38 KB |

### Quantile regression with ℓ1

| n, p | solver | time to 1e-4 | time to 1e-6 | time to 1e-8 | bytes/iteration |
|---|---|---:|---:|---:|---:|
| 500, 100 | ChambollePock (ProximalAlgorithms) | 37.7 ms | 245 ms | 1.62 s | 9.1 KB |
| 500, 100 | HiGHS simplex | 130 ms | 133 ms | 134 ms | |
| 2000, 200 | ChambollePock (ProximalAlgorithms) | — (1.1 after 60 s) | — | — | 34 KB |
| 2000, 200 | HiGHS simplex | 1.70 s | 1.79 s | 1.79 s | |

### TV denoising

| n | solver | time to 1e-4 | time to 1e-6 | time to 1e-8 | bytes/iteration |
|---:|---|---:|---:|---:|---:|
| 1000 | AccProxGrad on the dual (ProximalSolvers) | 1.12 ms | — (2.3e-6 at best) | — | 0 |
| 1000 | ChambollePock, VuCondat (ProximalAlgorithms) | 8.3 ms | 19.5 ms | 32 ms | 24 KB |
| 10000 | AccProxGrad on the dual (ProximalSolvers) | 20.1 ms | — (1.5e-5 at best) | — | 0 |
| 10000 | ChambollePock, VuCondat (ProximalAlgorithms) | 143 ms | 505 ms | 505 ms | 240 KB |

## What the measurements show

1. **ProximalAlgorithms can't use `LogDetLoss`'s convention of `f = Inf` off the domain.** Its backtracking (`backtrack_stepsize!` in `src/utilities/fb_tools.jl`) accepts a trial point when `f(z) > f_model + tol` fails, with `tol = 10 eps (1 + |f(z)|)`. When `f(z) = Inf`, `tol = Inf` and the test is `Inf > Inf`, which is false, so the infeasible point is accepted. From then on the iterates are NaN, and the stopping test `‖res‖/γ ≤ tol` never passes. That is why all five forward–backward-type solvers end at NaN on the graphical lasso, from any first step.
2. **`LiLin` is unusable as released.**
   - Its fallback branch (`src/algorithms/li_lin.jl`) evaluates `value_and_gradient(iter.f, x)` with an undefined `x`, so the first rejected extrapolation throws `UndefVarError` (D-trace above).
   - It takes a fixed step `1/L` and has no line search, so it can't run on `LogDetLoss` (`L = Inf`).
   - Where it works it is 9× slower than `ProxGrad` (IHT least squares).
3. **Allocations.** ProximalSolvers' iterations allocate nothing. ProximalAlgorithms':
   - allocates a gradient array per evaluation (its `value_and_gradient` interface);
   - its L-BFGS methods allocate up to 13 MB per iteration at p = 200;
   - its primal–dual methods allocate in the conjugate prox, `prox_conjugate!` without a buffer.
4. **The nonconvex problems are a wash.**
   - On IHT least squares, `ZeroFPR` is 15% faster than `ProxGrad`, and the others are slower.
   - On the D-trace IHT, all of ProximalAlgorithms' solvers stop at a local minimum 7% worse.
   - On MCP, `PANOC` finds the best local minimum, but takes 22× the time of coordinate descent, which ends 0.03% above it.
5. **The primal–dual methods** have fixed steps and no restarts.
   - On quantile regression they reach 1e-6 in 245 ms at n = 500, but don't converge in 60 s at n = 2000, where the LP takes 1.8 s.
   - On TV denoising they reach 1e-8 in 32 ms (n = 1000), where the dual formulation with ProximalSolvers' stopping rules stalls near 1e-5 in the primal.
   - Their default step sizes call `opnorm(L)`, which throws for a sparse `L` (`2-norm not yet implemented for sparse matrices`), so the steps were passed explicitly here.

## Decisions

- **#24 (Li–Lin):** points 1 and 2 are the structural reasons the issue asked for, so the monotone accelerated proximal gradient method is implemented natively: `MonotoneAccProxGradDescent`, PR #50.
- **#25 (PANOC, ZeroFPR, PANOCplus):**
  - Their main target here, the locally Lipschitz `LogDetLoss`, is exactly where ProximalAlgorithms fails (point 1), and `AccProxGradDescent` already solves it (56 ms to 1e-6 at p = 200).
  - On the nonconvex problems they bring no clear gain (point 4).
  - Recommendation: don't adopt ProximalAlgorithms' versions. Write a native PANOCplus only if a prototype beats `AccProxGradDescent` and the QUIC solver of #22 on the graphical lasso, keeping #25 open until then.
- **#23 (primal–dual):** the measurements meet the issue's conditions for a native solver (allocations, no restarts), but neither of its motivating problems needs one now:
  - quantile regression with ℓ1 is solved faster by the LP at these sizes;
  - 1-D TV gets ProximalBase's direct O(n) prox (ProximalBase #35).

  Recommendation: keep #23 open, and implement it with PDLP-style restarts and adaptive steps (the failure mode measured above) when a problem needs a first-order primal–dual method at a scale where the LP is slow, such as a generalized lasso with a large design.
