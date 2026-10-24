# Sparse group lasso paths against R's SGL and sparsegl

Issue #60 adds the paths of the sparse group lasso at a fixed mixture α,
`Scaled(SparseGroupNormL2(α, 1 - α, groups), λ)`. This folder times them
against the two R packages that compute the same paths:

- [SGL](https://cran.r-project.org/package=SGL) (Simon, Friedman, Hastie and
  Tibshirani), version 1.3;
- [sparsegl](https://cran.r-project.org/package=sparsegl) (Liang, Cohen,
  Sólon Heinsfeld and McDonald), version 1.1.1.

All three fit

    (1/(2n)) ‖y - Xβ‖² + λ (α ‖β‖₁ + (1 - α) Σ_g √|g| ‖β_g‖₂)

on the same λ grid. The grid has 50 levels, geometric from ProximalSolvers'
exact λmax down to 0.01 λmax. SGL always centers X and y and fits an
intercept, so the problems are centered and the other two fit no intercept.
Neither R package standardizes.

## Running

```
julia --project=benchmark/sgl -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/sgl benchmark/sgl/problems.jl PROBLEMS          # writes the problems
julia --project=benchmark/sgl benchmark/sgl/run_julia.jl PROBLEMS/A 0.5 RESULTS/A/0.5 3
Rscript benchmark/sgl/run_r.R PROBLEMS/A 0.5 RESULTS/A/0.5 sparsegl 3     # also SGL, SGL-tight
julia --project=benchmark/sgl benchmark/sgl/summary.jl PROBLEMS RESULTS   # the table
```

`problems.jl PROBLEMS SMOKE` writes a small problem instead. The Julia
environment loads ProximalSolvers from this repository and ProximalBase from
its master through `[sources]`, so it needs Julia 1.11 or later.

## What is measured

- **The problems.** Problem A has n = 200, p = 1000 and groups of 10; problem B
  has n = 500, p = 5000 and groups of 20. Columns are Gaussian, correlated 0.3
  within a group. The true β has half the entries of the first 5 (A) or 8 (B)
  groups nonzero, and the signal-to-noise ratio is about 2.
- **The mixtures:** α = 0.05 (sparsegl's default), 0.5 and 0.95 (SGL's
  default).
- **The solvers, each at its defaults:**
  - ProximalSolvers: `solve_path` with `CDSolver()`, with the strong rule
    (`tol = 1e-7`), and again with `gapsafe = true`;
  - SGL: `thresh = 1e-3`, its default. Its paths solve every level from zero,
    so SGL-tight also runs it with `thresh = 1e-6` and `maxit = 10000`;
  - sparsegl: `eps = 1e-8`.
- **Time:** the whole path, as the median of 3 runs. SGL-tight runs once. The
  Julia time comes after a first run that compiles.
- **Objective:** for every level, each package's objective is computed by
  `summary.jl` with dense algebra and compared with the best of all packages at
  that level. The columns are the largest relative excess over the path and the
  number of levels where it exceeds 1e-6.

## Results

See the table in the pull request of issue #60 (Discovery, AMD EPYC 7513, one
thread per run).
