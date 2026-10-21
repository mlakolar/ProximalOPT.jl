# ProximalOPT.jl

[![CI](https://github.com/mlakolar/ProximalOPT.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/ProximalOPT.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/ProximalOPT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/ProximalOPT.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Proximal optimization solvers in Julia for problems of the form

```
minimize  f(x) + g(x)
```

where `f` is smooth and `g` has an easy proximal operator (lasso, group lasso,
nuclear-norm and graphical-lasso penalties, ...). The loss functions and
proximal operators come from
[ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl); ProximalOPT
provides the solvers that put them together.

## Installation

Requires Julia 1.10 or later. ProximalOPT is not registered; install it from
GitHub (ProximalBase is installed from the General registry):

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/ProximalOPT.jl")
```

## Example: lasso

Minimize `‖Y - Xβ‖² / 2 + λ‖β‖₁`:

```julia
using ProximalBase, ProximalOPT, LinearAlgebra, Random

Random.seed!(1)
n, p = 100, 50
X = randn(n, p)
β_true = [ones(5); zeros(p - 5)]
Y = X * β_true + 0.1 * randn(n)

f = LeastSquaresLoss(Y, X)   # f(β) = ‖Y - Xβ‖² / 2
g = ProxL1(10.0)             # g(β) = 10 ‖β‖₁

β = zeros(p)                 # starting point, overwritten with the solution
res = solve!(β, f, g, AccProxGradDescent())

res.minimum, res.iterations, count(!iszero, β)
```

`β` recovers the five nonzero coefficients and sets the rest to exactly zero.
Replace `AccProxGradDescent()` with `ProximalGradientDescent()` to use the
unaccelerated method.

## Example: graphical lasso by ADMM

Estimate a sparse precision matrix by minimizing
`tr(SΘ) - log det Θ + λ‖Θ‖₁`, where both terms have closed-form proximal
operators:

```julia
using ProximalBase, ProximalOPT, LinearAlgebra, Random

Random.seed!(1)
Z = randn(200, 10)
S = Symmetric(Z' * Z / 200)          # sample covariance

Θ = Matrix(1.0I, 10, 10)             # starting point
res = solve!(Θ, ProxGaussLikelihood(S), ProxL1(0.1), ADMMSolver(; ρ = 1.0))

res.converged, res.iterations
```

## Solvers

| Solver | Problem | Method |
|---|---|---|
| `ProximalGradientDescent()` | `f` smooth, `g` proximable | Proximal gradient with TFOCS backtracking (Parikh and Boyd, §4.2) |
| `AccProxGradDescent(; β = 0.5)` | `f` smooth, `g` proximable | Accelerated proximal gradient (Parikh and Boyd, §4.3) with backtracking and adaptive restart |
| `ADMMSolver(; ρ, α, abstol, reltol)` | `f`, `g` both proximable | Over-relaxed scaled ADMM (Boyd et al., 2011) |

All solvers are called the same way:

```julia
res = solve!(x, f, g, solver, ProximalOptions(; maxiter = 1000))
```

`solve!` overwrites `x` with the solution and returns a results object with
`minimizer`, `minimum`, `iterations`, convergence information and the number
of evaluations of `f` (`f_calls`), of its gradient (`grad_calls`) and of
proximal operators (`prox_calls`); printing it shows a summary.

## Options and stopping rules

`ProximalOptions` controls the iteration and call limits, tolerances and
tracing:

| Keyword | Default | Meaning |
|---|---|---|
| `maxiter` | `200` | Maximum number of iterations |
| `ftol` | `1e-6` | Relative change in the objective |
| `xtol` | `1e-8` | Relative change in the iterate |
| `gradtol` | `1e-5` | Optimality residual (FASTA; Goldstein, Studer and Baraniuk) |
| `time_limit` | `Inf` | Maximum run time in seconds |
| `f_calls_limit`, `grad_calls_limit`, `prox_calls_limit` | `0`, `0`, `0` | Maximum number of evaluations of `f`, of its gradient and of proximal operators; `0` means no limit |
| `show_trace`, `store_trace`, `extended_trace`, `printEvery` | `false`, `false`, `false`, `1` | Print or record progress |
| `callback` | `nothing` | Called every `printEvery` iterations; return `true` to stop |

The gradient solvers stop when any of the `ftol`, `xtol` or `gradtol` tests
passes. `ADMMSolver` uses its own primal and dual residual tolerances
(`abstol`, `reltol`) and ignores the other three, as well as `f_calls_limit`
and `grad_calls_limit`. The call limits are checked after each iteration, so a
count can exceed its limit by the evaluations of one iteration;
`res.call_limit_reached` tells whether a limit stopped the solver.

Every exported function and type has a docstring; use `?name` in the Julia
REPL, for example `?ProximalOptions`.

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

## License

MIT, see [LICENSE](LICENSE).
