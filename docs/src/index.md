# ProximalSolvers.jl

Solvers for problems of the form

```math
\underset{x}{\text{minimize}} \quad F(x) = f(x) + g(x),
```

with a loss ``f`` and a penalty or constraint ``g``: the lasso and its
relatives, sparse generalized linear models, the group lasso, nonconvex
penalties, low-rank matrix regression, the graphical lasso and other
problems of high-dimensional statistics. The functions ``f`` and ``g`` come
from [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl), which
supplies their values, gradients and proximal operators.

The core is coordinate descent, for a penalty that separates across
coordinates or groups of coordinates. On top of it come penalty paths with
screening, reweighting, exact refits on a support, and proximal Newton for
generalized linear models. For a penalty that does not separate, such as the
nuclear norm, there are proximal gradient methods and ADMM.

## Installation

```julia
using Pkg
Pkg.add("ProximalSolvers")
```

ProximalSolvers requires Julia 1.10 or later. It was called ProximalOPT before
its first release.

## A first example

The lasso, ``\|y - X\beta\|^2/(2n) + \lambda\|\beta\|_1``, by coordinate
descent, at one penalty level and along a path:

```@example index
using ProximalBase, ProximalSolvers, Random

rng = Xoshiro(1)
n, p = 200, 1000
X = randn(rng, n, p)
y = X[:, 1:5] * [2.0, -1.5, 1.0, -1.0, 0.5] + randn(rng, n)

f = (1 / n) * LeastSquaresLoss(y, X)            # ‖y - Xβ‖² / (2n)
λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))
β = SparseIterate(p)                            # the starting point, overwritten
res = solve!(β, f, NormL1(0.1λmax), CDSolver())
findall(!iszero, β)
```

```@example index
path = solve_path(f, NormL1(1.0), λmax .* 10 .^ range(0, -2; length = 50))
[count(!iszero, b) for b in path.solutions[1:5:end]]
```

Every solver is called the same way, `solve!(x, f, g, solver)`, and overwrites
the starting point `x` with the solution.

## Choosing a solver

| Problem | Solver | Method |
|:--------|:-------|:-------|
| ``g`` separable: ℓ1, elastic net, MCP, SCAD, bounds; or the group lasso. ``f`` least squares, a quadratic, a Kronecker quadratic, a square-root loss | [`CDSolver`](@ref) | [Coordinate descent](@ref), with momentum or Anderson acceleration |
| ``f`` a generalized linear loss (logistic, Poisson, Huber), ``g`` separable | [`ProxNewton`](@ref) | [Proximal Newton](@ref): coordinate descent on weighted least squares |
| ``f`` smooth, ``g`` with a proximal operator | [`AccProxGradDescent`](@ref), [`ProximalGradientDescent`](@ref) | [Proximal gradient methods](@ref) |
| the same, ``g`` nonconvex: the ℓ0 ball, MCP, SCAD, a rank constraint | [`MonotoneAccProxGradDescent`](@ref), [`ProximalGradientDescent`](@ref) | [Nonconvex penalties](@ref) |
| the same, with a sparse solution and many coordinates | [`ActiveAccProxGradDescent`](@ref) | [Active-set proximal gradient](@ref) |
| ``f`` and ``g`` both with proximal operators | [`ADMMSolver`](@ref) | [ADMM](@ref) |

Coordinate descent is usually the fastest when it applies: it exploits the
structure of the loss to make each coordinate step cost ``O(n)`` or less,
and the sparsity of the solution to skip most coordinates. The drivers built
on it compute

- the solutions along a decreasing sequence of penalty levels, screened by
  the strong rule ([`solve_path`](@ref), see
  [Penalty paths and screening](@ref));
- estimators that alternate solves with updates of the penalty, such as the
  scaled lasso ([`reweight!`](@ref)), and refits on a support
  ([`refit!`](@ref)), see [Reweighting and refitting](@ref).

## The documentation

- [Calling the solvers](@ref) covers the calling convention, the options, the
  results and the stopping rules.
- The method pages state each algorithm as implemented, with the derivations
  of its steps and stopping rules: [Proximal gradient methods](@ref),
  [Active-set proximal gradient](@ref), [ADMM](@ref), and for coordinate
  descent [The method](@ref "Coordinate descent"), [The losses](@ref "Losses for coordinate descent"),
  [Penalty paths and screening](@ref), [Reweighting and refitting](@ref) and
  [Proximal Newton](@ref).
- [A model of your own](@ref "A coordinate descent model of your own") shows
  how to add a loss to coordinate descent.
- The examples are thirteen complete programs, each checking its result.
  The same scripts are in the [`examples/`](https://github.com/mlakolar/ProximalSolvers.jl/tree/master/examples)
  folder of the repository, and the test suite runs them.

The functions, their proximal operators and the sparse iterates are documented
with ProximalBase, at
[mlakolar.github.io/ProximalBase.jl](https://mlakolar.github.io/ProximalBase.jl/dev/).

## Related packages

ProximalSolvers is one of five Julia packages for high-dimensional statistics,
three for estimation and inference and two for the optimization beneath them:

| Package | |
|:--------|:--|
| [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl) | the lasso and its paths; the group, square-root, scaled and heteroscedastic lasso; varying-coefficient models |
| [CovarianceSelection.jl](https://github.com/mlakolar/CovarianceSelection.jl) | Gaussian graphical models: the graphical lasso, CLIME and ACLIME, precision matrices by the scaled lasso, the fused graphical lasso, and the difference of two precision matrices |
| [DiffPrecTest.jl](https://github.com/mlakolar/DiffPrecTest.jl) | confidence intervals and tests for the difference of two precision matrices |
| **ProximalSolvers.jl** (this package) | the solvers: coordinate descent with penalty paths, screening and active sets; proximal gradient methods, proximal Newton and ADMM |
| [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) | the building blocks: losses, penalties and constraints with their gradients and proximal operators, and sparse iterates; [documentation](https://mlakolar.github.io/ProximalBase.jl/dev/) with the mathematics |

Most users need one of the first three. To solve a problem of your own, use
ProximalSolvers; to write your own algorithm, ProximalBase.
