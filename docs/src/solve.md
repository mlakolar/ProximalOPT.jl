# Calling the solvers

## The call

Every solver minimizes ``f(x) + g(x)`` through [`solve!`](@ref):

```julia
res = solve!(x, f, g, solver)            # CDSolver, ProxNewton
res = solve!(x, f, g, solver, options)   # the proximal gradient solvers and ADMM
```

`x` is the starting point, and `solve!` overwrites it with the solution. The
loss `f` and the penalty `g` are functions of ProximalBase. Each solver asks
for some of their oracles, and for an iterate of a certain kind:

| Solver | ``f`` | ``g`` | `x` |
|:-------|:------|:------|:----|
| [`ProximalGradientDescent`](@ref), [`AccProxGradDescent`](@ref) | smooth: `issmooth(f)` | proximable: `isproximable(g)` | any array that `f` and `g` take, a `SparseIterate`, or an `AtomIterate` with the groups of a `GroupNormL2` |
| [`MonotoneAccProxGradDescent`](@ref) | smooth | proximable | any array that `f` and `g` take, a `SparseIterate` or a `SymmetricSparseIterate` |
| [`ActiveAccProxGradDescent`](@ref) | smooth, with `restrict` | separable or `GroupNormL2`, with `restrict` and `kkt_violation` | a vector, a `SparseIterate` or an `AtomIterate` |
| [`ADMMSolver`](@ref) | proximable | proximable | an `Array` |
| [`CDSolver`](@ref) | a loss with a coordinate descent model, see [Losses for coordinate descent](@ref) | separable, or a group penalty (`GroupNormL2`, `GroupNormLinf`, ...) | a `SparseIterate`, a `SymmetricSparseIterate`, an `AtomIterate` for a group penalty, or a vector |
| [`ProxNewton`](@ref) | a `GLMLoss`, or a `LogDetLoss` | separable, or a `NormL1` with a `LogDetLoss` | a `SparseIterate`; for a `LogDetLoss`, a `SymmetricSparseIterate` or a symmetric `Matrix` |

The iterates are those of ProximalBase: a
[`SparseIterate`](@extref ProximalBase.SparseIterate) stores the entries that
have been nonzero and gives ``O(1)`` access to any entry; a
[`SymmetricSparseIterate`](@extref ProximalBase.SymmetricSparseIterate) is a
sparse symmetric matrix whose coordinates are the entries of its lower
triangle; an [`AtomIterate`](@extref ProximalBase.AtomIterate) is a vector
whose coordinates are groups of entries. A solver that takes a dense vector
for coordinate descent or the active-set method converts it to one of these,
and copies the solution back.

The solvers compute in the element type of `x`, `Float64` or `Float32`. In
`Float32`, tolerances below its precision, `eps(Float32) ≈ 1.2e-7`, cannot be
met.

## Options

[`ProximalOptions`](@ref) holds the iteration and call limits, the tolerances
and the tracing of the proximal gradient solvers, which pass them on to each
inner solve of the active-set method:

| Keyword | Default | Meaning |
|:--------|:--------|:--------|
| `maxiter` | `200` | the largest number of iterations |
| `ftol` | `1e-6` | relative change of the objective |
| `xtol` | `1e-8` | relative change of the iterate |
| `gradtol` | `1e-5` | the optimality residual of FASTA |
| `allow_f_increases` | `false` | keep going when the objective increases |
| `time_limit` | `Inf` | the longest run time, in seconds |
| `f_calls_limit`, `grad_calls_limit`, `prox_calls_limit` | `0` | the largest number of evaluations of ``f``, of its gradient and of proximal operators; `0` is no limit |
| `store_trace`, `show_trace`, `extended_trace`, `printEvery` | `false`, `false`, `false`, `1` | record or print the progress |
| `callback` | `nothing` | called every `printEvery` iterations; returning `true` stops |
| `gaptol`, `gapevery` | `0`, `10` | with `gaptol > 0`, stop when the relative duality gap, computed every `gapevery` iterations, is at most `gaptol` (see [Duality gaps](@ref)) |

The tests of `ftol`, `xtol` and `gradtol` are those of
[Proximal gradient methods](@ref "Stopping rules"); the solver stops when any
of them passes. [`ADMMSolver`](@ref) has its own tolerances and uses only
`maxiter`, `time_limit`, `prox_calls_limit`, `show_trace`, `printEvery`,
`gaptol` and `gapevery`.
[`CDSolver`](@ref) and [`ProxNewton`](@ref) take their settings as keywords
of their constructors instead: `CDSolver(; maxpasses, tol, order, relative,
anderson, momentum, rng, gaptol, gapsafe)` and `ProxNewton(; inner, maxiter, tol, β)`.

The call limits are checked after each iteration, which always finishes, so a
count can exceed its limit by the evaluations of one iteration.

## Results

| Solver | Result | Fields |
|:-------|:-------|:-------|
| [`ProximalGradientDescent`](@ref), [`AccProxGradDescent`](@ref), [`MonotoneAccProxGradDescent`](@ref) | [`OptimizationResults`](@ref ProximalSolvers.OptimizationResults) | `minimizer`, `minimum`, `iterations`, the flags `x_converged`, `f_converged`, `grad_converged` with the final residuals, `f_increased`, `iteration_converged` (the iteration limit was reached), `call_limit_reached`, `backtrack_steps`, `f_calls`, `grad_calls`, `prox_calls`, `trace`, `gap` |
| [`ActiveAccProxGradDescent`](@ref) | [`ActiveSetResults`](@ref ProximalSolvers.ActiveSetResults) | `minimizer`, `minimum`, `outer_iterations`, `inner_iterations`, `converged`, `inner_converged`, `max_violation`, `active_set_size` |
| [`ADMMSolver`](@ref) | [`ADMMResults`](@ref ProximalSolvers.ADMMResults) | `minimizer`, `minimum`, `iterations`, `converged`, the residuals `r_norm`, `s_norm` and their tolerances `eps_pri`, `eps_dual`, the scaled dual variable `u` and the final `ρ`, the counts of calls, `gap` |
| [`CDSolver`](@ref), [`ProxNewton`](@ref) | [`CDResults`](@ref) | `minimizer`, `minimum`, `passes`, `converged`, `maxchange`, `tolerance`, `kkt_checks`, `gap` |
| [`solve_path`](@ref) | [`PathResults`](@ref) | per level: `λ`, `solutions`, `values`, `passes`, `converged`, `kkt_checks`, `stages` |
| [`reweight!`](@ref), `solve!` with [`LLA`](@ref) | [`ReweightResults`](@ref) | `minimizer`, `penalty`, `next`, `iterations`, `converged`, `change`, `passes`, `cd_converged` |

`minimum` is ``f + g`` at the minimizer, and `gap` the relative duality gap
there when it was computed, `NaN` otherwise. Printing a result shows a summary.

## Duality gaps

For a loss of a linear map, ``f(x) = F(Ax)`` with
``F(\eta) = \sum_i \phi_i(\eta_i)``, and a convex penalty ``g``, every dual
point ``\theta`` bounds the minimum of ``P = f + g`` from below:

```math
D(\theta) = -F^*(\theta) - g^*(-A^\top\theta) \le \min P \le P(x).
```

[`duality_gap`](@ref) builds the dual point from the gradient,
``\theta = t\,\nabla F(Ax)``, with the largest ``t \in (0, 1]`` that puts
``-A^\top\theta`` in the domain of ``g^*``, and returns ``P(x) - D(\theta)``:
an upper bound on ``P(x) - \min P`` that is zero at the solution. It knows
least squares and the logistic, Poisson and Huber losses, weighted, scaled or
through a model with an intercept, with `NormL1`, `NormL1Nonneg`, `ElasticNet`
or `GroupNormL2`; and the graphical lasso, a `LogDetLoss` with a `NormL1`,
whose dual point comes from the inverse of ``x``.

The relative gap ``(P(x) - D(\theta))/|P(x)|`` is a tolerance that means the
same for every problem. With `gaptol > 0`, [`CDSolver`](@ref) computes it,
at the cost of a full gradient, where its rule on the changes would stop, and
goes on with a tenfold smaller `tol` until the gap is at most `gaptol`. With
`gapsafe = true`, each gap also screens the coordinates by the gap-safe rule
of Fercoq, Gramfort and Salmon (*Mind the duality gap: safer rules for the
lasso*, 2015): a coordinate it proves zero at the solution is set to zero and
left out of the full passes. `ProximalOptions(; gaptol, gapevery)` stops
[`ProximalGradientDescent`](@ref), [`AccProxGradDescent`](@ref) and
[`ADMMSolver`](@ref) on the gap, computed every `gapevery` iterations, in
place of their other tests. For other losses and penalties `gaptol` is
ignored, with a warning. For the losses without a gap, such as the quadratics,
[`kkt_residual`](@ref) measures optimality on the scale of the stopping rule of
coordinate descent.

## Traces and callbacks

With `store_trace = true` the proximal gradient solvers record an
[`OptimizationState`](@ref) per iteration in `res.trace`: the iteration, the
objective `fg_x`, the relative change of the iterate `dxx` and the inverse step
size `L`. `extended_trace = true` also stores the iterate and the gradient.
`show_trace = true` prints the same every `printEvery` iterations. A
`callback` receives the trace, or the current state when the trace is not
stored, and stops the solver by returning `true`.

For the penalty paths of coordinate descent, `solve_path(...; callback)` calls
`callback(k, λ, x, m)` after each level instead; see
[Penalty paths and screening](@ref).
