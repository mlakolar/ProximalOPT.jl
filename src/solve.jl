
# remove the zero entries or groups from the stored ones of a sparse iterate, so
# that products by it touch only its support; nothing to do for an Array
_drop!(x::AbstractArray) = x
_drop!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}) = dropzeros!(x)

"""
    solve!(x, f, g, method::ProximalSolver, options = ProximalOptions())

Minimize `f(x) + g(x)` starting from `x`, and overwrite `x` with the solution.

For [`ProximalGradientDescent`](@ref) and [`AccProxGradDescent`](@ref), `f` is
a smooth function of ProximalBase (`issmooth(f)`) and `g` a proximable one
(`isproximable(g)`), and `x` is an array of any shape that they accept, such as
the `p × k` coefficient matrix of a `LeastSquaresLoss` with `k` responses and a
`NuclearNorm` penalty; the result is an [`OptimizationResults`](@ref). There `x`
may also be a `SparseIterate`, or for the group lasso `GroupNormL2` an
`AtomIterate` that shares the groups of `g` (`AtomIterate(x, g.groups)`): the
iterates keep its type, zeros are dropped after each step, and products by the
iterates touch only their stored entries or active groups. For
[`ActiveAccProxGradDescent`](@ref), `x` is a vector, a `SparseIterate` or an
`AtomIterate`, and the result is an `ActiveSetResults`. For
[`ADMMSolver`](@ref), both `f` and `g` are proximable and the result is an
`ADMMResults`. For [`CDSolver`](@ref) and [`ProxNewton`](@ref), see their
documentation.

# Example
```julia
using ProximalBase, ProximalOPT
A, b = [2. 0.; 0. 1.], [-1., -1.]
x = zeros(2)
res = solve!(x, QuadraticFunction(A, b), NormL1(0.1), AccProxGradDescent())
```
"""
function solve!(
    initial_x::AbstractArray,
    f::AbstractFunction,
    g::AbstractFunction,
    method::ProximalSolver,
    options::ProximalOptions = ProximalOptions()
    )

  issmooth(f) || throw(ArgumentError("$(summary(method)) needs a smooth f (issmooth(f)), but f is a $(typeof(f))."))
  isproximable(g) || throw(ArgumentError("$(summary(method)) needs a proximable g (isproximable(g)), but g is a $(typeof(g))."))
  t0 = time()
  state = initial_state(method, options, f, g, initial_x)

  tr = OptimizationTrace{typeof(method)}()
  tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback !== nothing
  stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
  f_limit_reached, grad_limit_reached, prox_limit_reached = false, false, false
  x_converged, fg_converged, grad_converged, f_increased = false, false, false, false

  converged = false
  iteration = 0

  options.show_trace && print_header(method, options)

  while !converged && !stopped && iteration < options.maxiter
    iteration += 1

    update_state!(f, g, state, method)

    x_converged, fg_converged,
    grad_converged, converged, f_increased = convergence_assessment(state, f, g, options)

    if tracing
      # update trace; callbacks can stop routine early by returning true
      stopped_by_callback = trace!(tr, f, g, state, iteration, method, options)
    end

    stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
    f_limit_reached = options.f_calls_limit > 0 && state.f_calls >= options.f_calls_limit
    grad_limit_reached = options.grad_calls_limit > 0 && state.grad_calls >= options.grad_calls_limit
    prox_limit_reached = options.prox_calls_limit > 0 && state.prox_calls >= options.prox_calls_limit

    if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
        stopped_by_time_limit || f_limit_reached || grad_limit_reached || prox_limit_reached
        stopped = true
    end
  end # while

  # the residuals are stored as Float64, like the tolerances, whatever the element type
  return OptimizationResults(
                method,
                state.x,
                state.f_x + state.g_x,
                iteration,
                iteration == options.maxiter,
                x_converged,
                options.xtol,
                Float64(norm_diff(state.x, state.x_previous, 2.)),
                fg_converged,
                options.ftol,
                Float64(f_residual(state.f_x + state.g_x, state.f_x_previous + state.g_x_previous, options.ftol)),
                grad_converged,
                options.gradtol,
                Float64(g_residual(state)),
                f_increased,
                tr,
                state.backtrack_steps,
                state.f_calls,
                state.grad_calls,
                state.prox_calls,
                f_limit_reached || grad_limit_reached || prox_limit_reached)

end
