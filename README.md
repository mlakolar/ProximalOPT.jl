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
[ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl) (0.5); ProximalOPT
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
g = NormL1(10.0)             # g(β) = 10 ‖β‖₁

β = zeros(p)                 # starting point, overwritten with the solution
res = solve!(β, f, g, AccProxGradDescent())

res.minimum, res.iterations, count(!iszero, β)
```

`β` recovers the five nonzero coefficients and sets the rest to exactly zero.
Replace `AccProxGradDescent()` with `ProximalGradientDescent()` to use the
unaccelerated method. When `p` is large and the solution sparse,
`ActiveAccProxGradDescent()` is usually much faster: it runs the accelerated
method on an active set of coordinates, which it grows by the coordinates that
violate the optimality conditions the most: by 100 at a time, or by as many as
are nonzero if that is more. Its starting point may be a vector, as here,
or a `SparseIterate`. The other solvers take a `SparseIterate` too, and then
multiply `X` by their sparse iterates over the nonzero entries only.

## Example: group lasso

Minimize `‖Y - Xβ‖² / 2 + λ Σₖ √|Gₖ| ‖β[Gₖ]‖₂` over groups `Gₖ` of
coefficients that enter or leave the model together, such as the dummy
variables of a factor:

```julia
using ProximalBase, ProximalOPT, LinearAlgebra, Random

Random.seed!(1)
n = 100
groups = [1:3, 4:5, 6:10, 11:12, [13, 15], [14]]   # sizes 3, 2, 5, 2, 2 and 1
X = randn(n, 15)
β_true = [ones(3); zeros(2); ones(5); zeros(5)]
Y = X * β_true + 0.1 * randn(n)

f = LeastSquaresLoss(Y, X)
g = GroupNormL2(10.0, groups)    # weights √|Gₖ| by default

β = zeros(15)
res = solve!(β, f, g, AccProxGradDescent())

[norm(β[G]) for G in groups]
```

The first and third groups are selected, and the other four are exactly zero.
`GroupNormL2(λ, groups, w)` takes other weights, and coordinates in no group
are not penalized. When the groups cover all coefficients, as here,
`AtomIterate(β, g.groups)` is a vector whose coordinates are the groups: the solvers
take it as a starting point and multiply by it over the nonzero groups only,
and `ActiveAccProxGradDescent()` adds and removes whole groups, as it also does
from a dense starting point.

## Example: low-rank multi-response regression

Regress `k` responses on the same `p` predictors, `Y = XB + E`, with a `p × k`
coefficient matrix `B` of low rank, by minimizing `‖Y - XB‖²_F / 2 + λ‖B‖_*`.
The nuclear norm `‖B‖_*`, the sum of the singular values of `B`, sets the small
singular values to exactly zero, as the ℓ1 norm of the lasso does with the small
coefficients:

```julia
using ProximalBase, ProximalOPT, LinearAlgebra, Random

Random.seed!(1)
n, p, k = 200, 20, 10
X = randn(n, p)
B_true = randn(p, 2) * randn(2, k)    # rank 2
Y = X * B_true + 0.5 * randn(n, k)

σ = norm(Y - X * (X \ Y)) / sqrt((n - p) * k)   # noise level, from least squares
λ = σ * opnorm(X) * (sqrt(p) + sqrt(k))

f = LeastSquaresLoss(Y, X)            # f(B) = ‖Y - XB‖²_F / 2
g = NuclearNorm(λ)                    # g(B) = λ ‖B‖_*

B = zeros(p, k)                       # starting point, overwritten with the solution
res = solve!(B, f, g, AccProxGradDescent())

rank(B; atol = 1e-6), norm(B - B_true) / norm(B_true)
```

`B` has rank 2, as `B_true`, and relative error 0.029; least squares, `X \ Y`,
has rank 10 and relative error 0.026. As here, the starting point of
`ProximalGradientDescent()` and `AccProxGradDescent()` may be a matrix, or an
array of any shape that `f` and `g` accept.

The rule for `λ` makes the penalty dominate the noise in the gradient of `f` at
`B_true`, which is `-X'E`: for Gaussian noise with standard deviation `σ`, its
operator norm `‖X'E‖`, the largest singular value, is at most about
`σ‖X‖(√p + √k)`. The penalty also shrinks the singular values that it keeps, so
a smaller `λ` has a smaller error but keeps more of them: `λ = 50`, close to
`‖X'E‖ = 49.7`, still gives rank 2, with relative error 0.022, and `λ = 20`
gives rank 7 and relative error 0.016. For `λ ≥ ‖X'Y‖`, `B = 0`.

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
res = solve!(Θ, LogDetLoss(S), NormL1(0.1), ADMMSolver(; ρ = 1.0))

res.converged, res.iterations
```

## Solvers

| Solver | Problem | Method |
|---|---|---|
| `ProximalGradientDescent()` | `f` smooth, `g` proximable | Proximal gradient with TFOCS backtracking (Parikh and Boyd, §4.2) |
| `AccProxGradDescent(; β = 0.5)` | `f` smooth, `g` proximable | Accelerated proximal gradient (Parikh and Boyd, §4.3) with backtracking and adaptive restart |
| `ActiveAccProxGradDescent(; β, max_add, growth, max_outer)` | `f` smooth, `g` separable (`NormL1`, ...) or `GroupNormL2`, sparse solution | Active set: adds the coordinates (whole groups, for the group lasso) that violate the optimality conditions, the largest violations first, about `max(max_add, (growth - 1) s)` entries when `s` are nonzero, and runs `AccProxGradDescent` on the active coordinates only |
| `ADMMSolver(; ρ, α, abstol, reltol, adaptive)` | `f`, `g` both proximable | Over-relaxed scaled ADMM (Boyd et al., 2011), optionally with residual balancing of `ρ`; warm starts from the scaled dual variable and `ρ` of a previous result |

All solvers are called the same way:

```julia
res = solve!(x, f, g, solver, ProximalOptions(; maxiter = 1000))
```

`solve!` overwrites `x` with the solution and returns a results object with
`minimizer`, `minimum`, `iterations`, convergence information and the number
of evaluations of `f` (`f_calls`), of its gradient (`grad_calls`) and of
proximal operators (`prox_calls`); printing it shows a summary. The solvers
work in the element type of `x`, `Float64` or `Float32`.

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

## Benchmark

[`benchmark/`](benchmark) times the solvers on the lasso against coordinate
descent ([CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl)),
glmnet and Lasso.jl; [benchmark/README.md](benchmark/README.md) explains how to
run it and summarizes the results.

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
