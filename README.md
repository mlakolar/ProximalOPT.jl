# ProximalSolvers.jl

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/ProximalSolvers.jl/dev/)
[![CI](https://github.com/mlakolar/ProximalSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/ProximalSolvers.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/ProximalSolvers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/ProximalSolvers.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Coordinate descent and proximal solvers in Julia for problems of the form

```
minimize  f(x) + g(x)
```

The core is coordinate descent. It handles a penalty `g` that separates across
coordinates or groups of coordinates (lasso, elastic net, MCP, SCAD, group
lasso, box constraints) and a loss `f` whose coordinates are cheap to update
(least squares, quadratics, Kronecker-structured quadratics of matrices,
generalized linear losses). On top of it come paths of penalty levels with
screening, reweighting and exact refits.

It also has the other proximal solvers, for a smooth `f` and a `g` with an easy
proximal operator: proximal gradient descent (plain, accelerated, or on an
active set) and ADMM. They cover the nuclear norm, the graphical lasso and
other problems that coordinate descent does not fit.

The losses, penalties and proximal operators come from
[ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) (0.5), and the
next releases of CoordinateDescent, CovarianceSelection and DiffPrecTest run on
these solvers.

ProximalSolvers is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|---|---|
| [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| **ProximalSolvers.jl** (this package) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.

## Installation

Requires Julia 1.10 or later. Install it from the General registry, or the
development version from GitHub:

```julia
using Pkg
Pkg.add("ProximalSolvers")
Pkg.add(url = "https://github.com/mlakolar/ProximalSolvers.jl")   # development version
```

ProximalSolvers was called ProximalOPT before its first release.

## Usage

Every solver is called the same way: `solve!(x, f, g, solver)` minimizes
`f(x) + g(x)` from the starting point `x`, overwrites `x` with the solution,
and returns a summary of the run. The lasso in glmnet's scaling,
`‖y - Xβ‖² / (2n) + λ‖β‖₁`, at one penalty level and along a path:

```julia
using ProximalBase, ProximalSolvers

n, p = 200, 1000
X = randn(n, p)
y = X[:, 1:5] * ones(5) + randn(n)

f = (1 / n) * LeastSquaresLoss(y, X)
λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))   # ‖X'y‖∞ / n
β = SparseIterate(p)
res = solve!(β, f, NormL1(0.1λmax), CDSolver())
path = solve_path(f, NormL1(1.0), λmax .* 10 .^ range(0, -2; length = 50))
```

The [examples](examples) are complete programs for the problems below, each
checking its own result.

## Solvers

| Solver | Problem | Method |
|---|---|---|
| `CDSolver(; maxpasses, tol, order, relative, anderson, momentum, gaptol, gapsafe)` | `f` least squares, a quadratic, a Kronecker quadratic, a square-root loss or a GLM; `g` separable or a group penalty (`GroupNormL2`, `GroupNormLinf`) | [Coordinate descent](https://mlakolar.github.io/ProximalSolvers.jl/dev/coordinate_descent/) with full and active passes, in random order with momentum or in cyclic order with Anderson acceleration, and a scale-free stopping rule or a duality gap |
| `ProxNewton(; inner, maxiter, tol, β)` | `f` a `GLMLoss` (logistic, Poisson, Huber), `g` separable; or `f` a `LogDetLoss`, `g` a `NormL1` (the graphical lasso) | [Proximal Newton](https://mlakolar.github.io/ProximalSolvers.jl/dev/prox_newton/): for a GLM glmnet's IRLS, coordinate descent on the weighted least squares model of `f`, then backtracking; for the graphical lasso QUIC, which stops on the duality gap |
| `ProximalGradientDescent(; normalized, debias)` | `f` smooth, `g` proximable, possibly nonconvex | [Proximal gradient](https://mlakolar.github.io/ProximalSolvers.jl/dev/proximal_gradient/) with the backtracking of TFOCS; with `g = IndBallL0(s)` it is iterative hard thresholding, and the options take normalized steps and refit a support that stopped changing |
| `AccProxGradDescent(; β = 0.5, α = 0.9)` | `f` smooth, `g` proximable | Accelerated proximal gradient with backtracking that also lets the step grow, and adaptive restart |
| `MonotoneAccProxGradDescent(; β, α, δ = 1e-3)` | `f` smooth, `g` proximable, possibly nonconvex (`IndBallL0`, `MCP`, ...) | [Monotone accelerated proximal gradient](https://mlakolar.github.io/ProximalSolvers.jl/dev/proximal_gradient/#Nonconvex-penalties) (Li and Lin): every limit point of the iterates is a critical point |
| `ActiveAccProxGradDescent(; β, α, max_add, growth, max_outer)` | `f` smooth, `g` separable or `GroupNormL2`, sparse solution | [Active set](https://mlakolar.github.io/ProximalSolvers.jl/dev/active_set/): the accelerated method on the coordinates, or whole groups, that violate the optimality conditions |
| `ADMMSolver(; ρ, α, abstol, reltol, adaptive)` | `f`, `g` both proximable | Over-relaxed scaled [ADMM](https://mlakolar.github.io/ProximalSolvers.jl/dev/admm/), optionally with residual balancing of `ρ`, and warm starts |

On top of coordinate descent, `solve_path` computes the solutions along a
decreasing sequence of penalty levels with warm starts and the sequential
strong rule, `reweight!` alternates solves with updates of the penalty (the
scaled lasso, adaptive weights, reweighted ℓ1), and `refit!` minimizes the loss
exactly on a support. A loss of your own joins coordinate descent as a
`CDModel` with a few functions; see `?CDModel`.

A duality gap certifies the accuracy of a solution. `duality_gap(f, g, x)`
bounds the distance of `f(x) + g(x)` to the minimum for least squares and the
logistic, Poisson and Huber losses with the ℓ1 norm, the elastic net or the
group lasso, and for the graphical lasso. With `gaptol`, `CDSolver` and the
solvers that take `ProximalOptions` stop on the relative gap, a tolerance that
means the same for every problem, and `CDSolver(; gapsafe = true)` adds
gap-safe screening. For the losses without a gap, `kkt_residual(f, g, x)`
measures optimality.

The losses coordinate descent knows, any of the first four scaled as in
`(1/n) * f`:

| Loss | Coordinate step | Use |
|---|---|---|
| `LeastSquaresLoss(y, X)`, `GLMLoss(ProximalBase.Squared(), y, X; weights, offset)` | O(n), residuals cached | lasso, weighted lasso; `cdmodel(f, x; intercept = true)` adds an unpenalized intercept |
| `QuadraticFunction(A, b)` | O(p), `Ax` cached | the lasso on `X'X/n` when `n ≫ p` |
| `KronQuadratic(A, B, C)`, `⟨Θ, AΘB⟩/2 + ⟨C, Θ⟩` | O(pq), `AΘB` cached | the D-trace loss of a symmetric `Θ`, inverse Hessians of Kronecker form |
| `GLMLoss` with `Logistic` or `Huber` | O(n), curvature bound | logistic and Huber lasso; `ProxNewton` is faster, and also fits Poisson and an intercept |
| `SqrtLeastSquaresLoss(y, X)`, `‖y - Xβ‖` | exact, O(n) | square-root lasso (with `NormL1`) |
| `SqrtQuadraticFunction(A, b, c)`, `√(x'Ax + 2b'x + c)` | exact, O(p), `Ax` cached | square-root lasso from `X'X/n`, `X'y/n` and `y'y/n`; the scaled lasso of one variable on the others from a covariance matrix (Sun and Zhang, 2013) |

With the random order, the default, the passes over the active set use
momentum. On CoordinateDescent.jl's benchmark it computes lasso paths 2.8 to
4.5 times faster and single solves at small penalties 1.3 to 2.2 times faster,
and on logistic lasso problems coordinate descent is 1.4 to 5.6 times faster
with it.

## Examples

The folder [`examples/`](examples) holds thirteen scripts, run by the test
suite and rendered in the documentation:

| Script | Problem |
|---|---|
| [`01_lasso.jl`](examples/01_lasso.jl) | the lasso by coordinate descent; its optimality conditions; an unpenalized intercept |
| [`02_lasso_path.jl`](examples/02_lasso_path.jl) | a lasso path with the strong rule; choosing λ on held-out data |
| [`03_accelerations.jl`](examples/03_accelerations.jl) | momentum and Anderson acceleration on correlated columns |
| [`04_penalties.jl`](examples/04_penalties.jl) | the elastic net, MCP and SCAD, continuation, the nonnegative lasso, bounds, unpenalized coordinates |
| [`05_group_lasso.jl`](examples/05_group_lasso.jl) | the group lasso by block coordinate descent, along a path, and by proximal gradient |
| [`06_glm.jl`](examples/06_glm.jl) | sparse logistic regression, Poisson regression with an exposure, Huber regression |
| [`07_sqrt_lasso.jl`](examples/07_sqrt_lasso.jl) | the square-root and scaled lasso; a sparse precision matrix column by column |
| [`08_refit_and_reweight.jl`](examples/08_refit_and_reweight.jl) | the least squares refit, the adaptive lasso, reweighted ℓ1 |
| [`09_differential_network.jl`](examples/09_differential_network.jl) | the difference of two precision matrices by the D-trace loss |
| [`10_proximal_gradient.jl`](examples/10_proximal_gradient.jl) | proximal gradient, accelerated and active-set solvers on the lasso; traces; `Float32` |
| [`11_low_rank_regression.jl`](examples/11_low_rank_regression.jl) | multi-response regression with the nuclear norm |
| [`12_graphical_lasso.jl`](examples/12_graphical_lasso.jl) | the graphical lasso by ADMM, with adaptive ρ and warm starts |
| [`13_custom_model.jl`](examples/13_custom_model.jl) | a coordinate descent model of your own, for a sparse design matrix |

## Documentation

The [documentation](https://mlakolar.github.io/ProximalSolvers.jl/dev/) states
each method as implemented, with the mathematics: the steps and their
derivations, the stopping rules, the options and the results. It covers the
proximal gradient methods, the active-set method and ADMM; for coordinate
descent, the coordinate steps of each loss, momentum and Anderson
acceleration, penalty paths and the strong rule, reweighting and refits, and
proximal Newton. Every exported function and type also has a docstring
(`?name` in the Julia REPL).

## Benchmark

[`benchmark/`](benchmark) times the solvers on the lasso against coordinate
descent ([CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl)),
glmnet and Lasso.jl; [benchmark/README.md](benchmark/README.md) explains how to
run it and summarizes the results. [`benchmark/engine/`](benchmark/engine)
compares the coordinate descent of ProximalSolvers, case by case, with the code it
replaces in CoordinateDescent,
[CovarianceSelection](https://github.com/mlakolar/CovarianceSelection.jl) (then
called CovSel) and [DiffPrecTest](https://github.com/mlakolar/DiffPrecTest.jl).

## References

- N. Parikh and S. Boyd. Proximal Algorithms. *Foundations and Trends in
  Optimization*, 1(3), 2014.
- S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein. Distributed
  Optimization and Statistical Learning via the Alternating Direction Method of
  Multipliers. *Foundations and Trends in Machine Learning*, 3(1), 2011.
- B. O'Donoghue and E. Candès. Adaptive Restart for Accelerated Gradient
  Schemes. *Foundations of Computational Mathematics*, 15, 2015.
- T. Goldstein, C. Studer and R. Baraniuk. A Field Guide to Forward-Backward
  Splitting with a FASTA Implementation. arXiv:1411.3406, 2014.
- S. Becker, E. Candès and M. Grant. Templates for Convex Cone Problems with
  Applications to Sparse Signal Recovery (TFOCS). *Mathematical Programming
  Computation*, 3, 2011.
- J. Friedman, T. Hastie and R. Tibshirani. Regularization Paths for
  Generalized Linear Models via Coordinate Descent. *Journal of Statistical
  Software*, 33(1), 2010.
- R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor and
  R. Tibshirani. Strong Rules for Discarding Predictors in Lasso-type Problems.
  *Journal of the Royal Statistical Society B*, 74(2), 2012.
- S. D. Zhao, T. T. Cai and H. Li. Direct Estimation of Differential Networks.
  *Biometrika*, 101(2), 2014.
- T. Sun and C.-H. Zhang. Sparse Matrix Inversion with Scaled Lasso. *Journal
  of Machine Learning Research*, 14, 2013.
- H. Li and Z. Lin. Accelerated Proximal Gradient Methods for Nonconvex
  Programming. *Advances in Neural Information Processing Systems* 28, 2015.
- T. Blumensath and M. E. Davies. Normalized Iterative Hard Thresholding:
  Guaranteed Stability and Performance. *IEEE Journal of Selected Topics in
  Signal Processing*, 4(2), 2010.

## License

MIT, see [LICENSE](LICENSE).
