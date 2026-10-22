# active-set version of the accelerated proximal gradient method: the active set
# is the set of stored entries of a SparseIterate; each outer iteration adds the
# coordinates that violate the optimality conditions and minimizes over the
# active set with AccProxGradDescent

"""
    ActiveAccProxGradDescent(; β = 0.5, max_add = typemax(Int), max_outer = 1000)

Active-set accelerated proximal gradient descent, for `f(x) + g(x)` with a
sparse minimizer, such as the lasso. The active set is the set of stored
entries of a `SparseIterate`. Each outer iteration

1. removes the coordinates that are zero from the active set (`dropzeros!`);
2. computes the full gradient `∇f(x)` and finds the coordinates outside the
   active set that violate the optimality conditions, `kkt_violation > 0`;
3. stops if there are none and `x` minimizes `f + g` over the active set, that
   is, the last inner solve converged (or `x = 0`); an inner solve stopped by
   `maxiter` is thus repeated;
4. adds the violators to the active set (`activate!`), only the `max_add`
   largest violations if there are more;
5. minimizes `f + g` over the active set with [`AccProxGradDescent`](@ref)`(; β)`,
   warm-started from the current values.

Step 5 runs on `restrict(f, S)` and `restrict(g, S)` from ProximalBase, a
problem in the active coordinates `S` only; step 2 is the only one that involves
all coordinates. `f` and `g` need `restrict`, `g` needs `kkt_violation`, and
`f` needs `value_and_gradient!` at a `SparseIterate`: in ProximalBase these are
`QuadraticFunction`, `L2Loss` and `LeastSquaresLoss`, with `ProxL1`,
`ProxGroupL2` or `ProxZero`. For the group lasso `ProxGroupL2`, all the
coordinates of a group at zero share the violation of the group, so a group
that violates the optimality conditions joins the active set whole, unless
`max_add` cuts it.

`max_outer` limits the number of outer iterations. The
[`ProximalOptions`](@ref) apply to each inner solve. Use with
[`solve!`](@ref), which returns an [`ActiveSetResults`](@ref).
"""
struct ActiveAccProxGradDescent{T<:AbstractFloat} <: ProximalSolver
  β::T            # backtracking factor of the inner AccProxGradDescent
  max_add::Int    # most violators added per outer iteration
  max_outer::Int  # outer iteration limit
end

Base.summary(::ActiveAccProxGradDescent) = "Active-Set Accelerated Proximal Gradient Descent"

function ActiveAccProxGradDescent(; β::AbstractFloat = 0.5, max_add::Integer = typemax(Int),
                                  max_outer::Integer = 1000)
  0 < β < 1 || error("β must be in (0, 1).")
  max_add > 0 || error("max_add must be positive.")
  max_outer > 0 || error("max_outer must be positive.")
  ActiveAccProxGradDescent(β, Int(max_add), Int(max_outer))
end

"""
    ActiveSetResults

Returned by [`solve!`](@ref) with an [`ActiveAccProxGradDescent`](@ref):
`minimizer` (the `x` passed to `solve!`), `minimum` (`f + g` at the
minimizer), `outer_iterations`, `inner_iterations` (summed over the inner
solves), `converged` (no coordinate outside the active set violates the
optimality conditions, and the last inner solve converged), `inner_converged`
(every inner solve converged), `max_violation` (the largest `kkt_violation`
outside the active set at the minimizer, `-Inf` if there is no such
coordinate) and `active_set_size`.
"""
struct ActiveSetResults{X,T}
    minimizer::X
    minimum::T
    outer_iterations::Int
    inner_iterations::Int
    converged::Bool
    inner_converged::Bool
    max_violation::T
    active_set_size::Int
end

Base.summary(::ActiveSetResults) = "Active-Set Accelerated Proximal Gradient Descent"
minimizer(r::ActiveSetResults) = r.minimizer
minimum(r::ActiveSetResults) = r.minimum
converged(r::ActiveSetResults) = r.converged

function Base.show(io::IO, r::ActiveSetResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Outer iterations: %d\n" r.outer_iterations
    @printf io " * Inner iterations: %d\n" r.inner_iterations
    @printf io " * Active set size: %d\n" r.active_set_size
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * largest KKT violation outside the active set: %.2e\n" r.max_violation
    @printf io "   * every inner solve converged: %s\n" r.inner_converged
    return
end

# x is the starting point and is overwritten with the solution; its stored
# entries are the active set
function solve!(
    x::SparseIterate{T, 1},
    f::DifferentiableFunction,
    g::ProximableFunction,
    method::ActiveAccProxGradDescent,
    options::ProximalOptions = ProximalOptions()
    ) where {T<:AbstractFloat}

  inner_method = AccProxGradDescent(; β = T(method.β))
  grad = Vector{T}(undef, length(x))
  violators, violations = Int[], T[]

  outer_iterations, inner_iterations = 0, 0
  solved = false              # x minimizes f + g over its stored entries
  inner_converged = true
  kkt_converged = false

  options.show_trace && @printf "Outer   Active    Added   Largest violation\n"

  while true
    dropzeros!(x)
    f_x = value_and_gradient!(f, grad, x)
    max_violation = _kkt_violators!(violators, violations, g, grad, x)

    # a minimizer over the active set that no coordinate outside it can improve
    # is a minimizer; so is x = 0 if no coordinate can improve it
    if isempty(violators) && (solved || nnz(x) == 0)
      kkt_converged = true
    end
    if kkt_converged || outer_iterations == method.max_outer
      return ActiveSetResults(x, f_x + value(g, x), outer_iterations, inner_iterations,
                              kkt_converged, inner_converged, max_violation, nnz(x))
    end
    outer_iterations += 1

    if length(violators) > method.max_add
      keep = partialsortperm(violations, 1:method.max_add; rev = true)
      violators = violators[keep]
    end
    for i in violators
      activate!(x, i)
    end

    if options.show_trace
      @printf "%5d   %6d   %6d   %14e\n" outer_iterations nnz(x) length(violators) max_violation
    end

    # minimize over the active set; the order of S matches the stored values
    k = nnz(x)
    S = x.nzval2ind[1:k]
    xS = x.nzval[1:k]
    res = solve!(xS, restrict(f, S), restrict(g, S), inner_method, options)
    copyto!(x.nzval, 1, xS, 1, k)

    inner_iterations += res.iterations
    solved = converged(res)
    inner_converged &= solved
  end
end

# a dense x is converted to a SparseIterate and overwritten with the solution
function solve!(
    x::Vector{T},
    f::DifferentiableFunction,
    g::ProximableFunction,
    method::ActiveAccProxGradDescent,
    options::ProximalOptions = ProximalOptions()
    ) where {T<:AbstractFloat}

  xs = SparseIterate(x)
  r = solve!(xs, f, g, method, options)
  copyto!(x, xs)
  ActiveSetResults(x, r.minimum, r.outer_iterations, r.inner_iterations,
                   r.converged, r.inner_converged, r.max_violation, r.active_set_size)
end

# Collect the coordinates outside the active set of x with a positive
# kkt_violation, and their violations; return the largest violation.
function _kkt_violators!(violators, violations, g, grad, x::SparseIterate{T}) where T
  empty!(violators)
  empty!(violations)
  max_violation = T(-Inf)
  for i in eachindex(grad)
    x.ind2nzval[i] == 0 || continue
    v = kkt_violation(g, grad, x, i)
    max_violation = max(max_violation, v)
    if v > 0
      push!(violators, i)
      push!(violations, v)
    end
  end
  max_violation
end
