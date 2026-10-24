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

Discovery, AMD EPYC 7513, one thread per run, Julia 1.13.2 and R 4.5.3
(ProximalSolvers #67, run `20261024T1221Z-ps-sgl-bench`).

**Median time of the whole path (s)**, over 3 runs. The `thresh = 1e-6` column is one run, and Julia's runs come after a compiling run.

| problem | α | ProximalSolvers | + gapsafe | sparsegl 1.1.1 | SGL 1.3 | SGL, thresh 1e-6 |
|---|---|---|---|---|---|---|
| A | 0.05 | **0.60** | 0.60 | 0.63 | 7.3 | 32 |
| A | 0.5 | **0.74** | 0.74 | 0.76 | 8.4 | 34 |
| A | 0.95 | **1.16** | 1.17 | 1.22 | 10.3 | 46 |
| B | 0.05 | **3.86** | 3.87 | 7.62 | 101 | 349 |
| B | 0.5 | **4.69** | 4.69 | 8.81 | 102 | 379 |
| B | 0.95 | **8.02** | 8.04 | 17.6 | 138 | 560 |

**Objective values.** At each level, the objective of each package's solution is compared with the best of all packages. Each cell is the largest relative excess along the path, with the number of the 50 levels where it exceeds 1e-6 in parentheses.

| problem | α | ProximalSolvers | + gapsafe | sparsegl | SGL | SGL, thresh 1e-6 |
|---|---|---|---|---|---|---|
| A | 0.05 | 6.0e-9 (0) | 5.2e-9 (0) | 2.3e-6 (5) | 2.6e-3 (34) | 4.9e-3 (36) |
| A | 0.5 | 4.3e-9 (0) | 4.3e-9 (0) | 3.4e-6 (7) | 2.8e-3 (32) | 5.5e-3 (35) |
| A | 0.95 | 1.3e-8 (0) | 1.3e-8 (0) | 8.1e-6 (13) | 3.2e-3 (33) | 9.1e-3 (29) |
| B | 0.05 | 2.0e-9 (0) | 4.0e-9 (0) | 8.5e-7 (0) | 1.0e-2 (36) | 1.9e-2 (39) |
| B | 0.5 | 5.9e-13 (0) | 5.9e-13 (0) | 1.3e-6 (2) | 1.6e-2 (37) | 2.3e-2 (40) |
| B | 0.95 | 1.1e-9 (0) | 6.7e-10 (0) | 3.1e-6 (7) | 2.0e-2 (36) | 3.0e-2 (39) |

- **Against sparsegl.** On A, ProximalSolvers takes 0.95–0.97 of sparsegl's time. On B it takes 0.46–0.53, and it is the most accurate everywhere: within 1.3e-8 of the best at every level, against sparsegl's up to 8e-6.
- **Against SGL.** SGL is 9–26× slower and 0.3–3% above the best objective at most levels. It solves every level from zero (`oneDim` resets `beta.old`). With `thresh = 1e-6` it is 3.5–4.4× slower again and no more accurate; we did not look into its stopping rule.
- **Gap-safe screening** neither helps nor costs here: after the strong rule, few groups are left for it to remove.
