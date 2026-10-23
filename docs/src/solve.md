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
| [`ActiveAccProxGradDescent`](@ref) | smooth, with `restrict` | separable or `GroupNormL2`, with `restrict` and `kkt_violation` | a vector, a `SparseIterate` or an `AtomIterate` |
| [`ADMMSolver`](@ref) | proximable | proximable | an `Array` |
| [`CDSolver`](@ref) | a loss with a coordinate descent model, see [Losses for coordinate descent](@ref) | separable, or `GroupNormL2` | a `SparseIterate`, a `SymmetricSparseIterate`, an `AtomIterate` for a group penalty, or a vector |
| [`ProxNewton`](@ref) | a `GLMLoss` | separable | a `SparseIterate` |

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

The tests of `ftol`, `xtol` and `gradtol` are those of
[Proximal gradient methods](@ref "Stopping rules"); the solver stops when any
of them passes. [`ADMMSolver`](@ref) has its own tolerances and uses only
`maxiter`, `time_limit`, `prox_calls_limit`, `show_trace` and `printEvery`.
[`CDSolver`](@ref) and [`ProxNewton`](@ref) take their settings as keywords
of their constructors instead: `CDSolver(; maxpasses, tol, order, relative,
anderson, momentum)` and `ProxNewton(; inner, maxiter, tol, β)`.

The call limits are checked after each iteration, which always finishes, so a
count can exceed its limit by the evaluations of one iteration.

## Results

| Solver | Result | Fields |
|:-------|:-------|:-------|
| [`ProximalGradientDescent`](@ref), [`AccProxGradDescent`](@ref) | [`OptimizationResults`](@ref ProximalSolvers.OptimizationResults) | `minimizer`, `minimum`, `iterations`, the flags `x_converged`, `f_converged`, `grad_converged` with the final residuals, `f_increased`, `iteration_converged` (the iteration limit was reached), `call_limit_reached`, `backtrack_steps`, `f_calls`, `grad_calls`, `prox_calls`, `trace` |
| [`ActiveAccProxGradDescent`](@ref) | [`ActiveSetResults`](@ref ProximalSolvers.ActiveSetResults) | `minimizer`, `minimum`, `outer_iterations`, `inner_iterations`, `converged`, `inner_converged`, `max_violation`, `active_set_size` |
| [`ADMMSolver`](@ref) | [`ADMMResults`](@ref ProximalSolvers.ADMMResults) | `minimizer`, `minimum`, `iterations`, `converged`, the residuals `r_norm`, `s_norm` and their tolerances `eps_pri`, `eps_dual`, the scaled dual variable `u` and the final `ρ`, the counts of calls |
| [`CDSolver`](@ref), [`ProxNewton`](@ref) | [`CDResults`](@ref) | `minimizer`, `minimum`, `passes`, `converged`, `maxchange`, `tolerance`, `kkt_checks` |
| [`solve_path`](@ref) | [`PathResults`](@ref) | per level: `λ`, `solutions`, `values`, `passes`, `converged`, `kkt_checks` |
| [`reweight!`](@ref) | [`ReweightResults`](@ref) | `minimizer`, `penalty`, `next`, `iterations`, `converged`, `change`, `passes`, `cd_converged` |

`minimum` is ``f + g`` at the minimizer. Printing a result shows a summary.

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
