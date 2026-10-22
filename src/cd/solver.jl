####################################
#
#  Coordinate descent: options, results and the passes
#
####################################

"""
    CDSolver(; maxpasses = 100_000, tol = 1e-7, order = :random, relative = true)

Coordinate descent for `f(x) + g(x)`, where `g` is separable
(`isseparable(g)`: `NormL1`, `ElasticNet`, `MCP`, `SCAD`, `IndBox`, …) or a
group penalty such as `GroupNormL2` on an `AtomIterate` with its groups, and
`f` is a loss whose structure [`solve!`](@ref) recognizes: `LeastSquaresLoss`,
a least-squares `GLMLoss`, `QuadraticFunction`, [`KronQuadratic`](@ref), or one
of these scaled, as in `(1/n) * LeastSquaresLoss(y, X)`.

Each pass visits the coordinates once, in random order (`order = :random`, from
the global random number generator) or in the order `1, 2, …`
(`order = :cyclic`). Passes over all coordinates alternate with passes over the
active ones (the stored entries of the iterate); the method stops after a full
pass followed by a pass in which no coordinate changes by more than `tol`, or
after `maxpasses` passes. With `relative`, the change of coordinate `k` is
measured as `|Δxₖ| √aₖ`, where `aₖ` is the curvature of `f` along the
coordinate, and compared with `tol` times a scale of the data, so that the
tolerance does not depend on the units of the data (for least squares,
`‖X[:, k]‖ √s` and `‖y‖ √s` for `f = s ‖y - Xβ‖² / 2`).
"""
struct CDSolver <: ProximalSolver
  maxpasses::Int
  tol::Float64
  random::Bool
  relative::Bool
end

function CDSolver(; maxpasses::Integer = 100_000, tol::Real = 1e-7, order::Symbol = :random,
                  relative::Bool = true)
  maxpasses >= 0 || throw(ArgumentError("maxpasses must be nonnegative, but it is $maxpasses."))
  tol >= 0 || throw(ArgumentError("tol must be nonnegative, but it is $tol."))
  order in (:random, :cyclic) || throw(ArgumentError("order must be :random or :cyclic, but it is :$order."))
  CDSolver(Int(maxpasses), Float64(tol), order === :random, relative)
end

Base.summary(::CDSolver) = "Coordinate Descent"

"""
    CDResults

Returned by [`solve!`](@ref) for a [`CDSolver`](@ref): the `minimizer` (the
iterate passed in), the `minimum` `f + g` at it, the number of `passes`, whether
the stopping rule was met (`converged`), the largest scaled change of a
coordinate in the last pass (`maxchange`) and the `tolerance` it was compared
with, and `kkt_checks`, the number of times the optimality conditions of the
coordinates left out by screening were checked.
"""
struct CDResults{X, T}
  minimizer::X
  minimum::T
  passes::Int
  converged::Bool
  maxchange::T
  tolerance::T
  kkt_checks::Int
end

Base.summary(::CDResults) = "Coordinate Descent"
minimizer(r::CDResults) = r.minimizer
minimum(r::CDResults) = r.minimum
iterations(r::CDResults) = r.passes
converged(r::CDResults) = r.converged

function Base.show(io::IO, r::CDResults)
  @printf io "Results of Optimization Algorithm\n"
  @printf io " * Algorithm: %s\n" summary(r)
  @printf io " * Minimum: %e\n" r.minimum
  @printf io " * Passes: %d\n" r.passes
  @printf io " * Convergence: %s\n" r.converged
  @printf io "   * largest change in the last pass %.2e (tolerance %.2e)\n" r.maxchange r.tolerance
  r.kkt_checks > 0 && @printf io " * Optimality checks after screening: %d\n" r.kkt_checks
  return
end

# the state of one run: the coordinate order and what the passes report
mutable struct _CDState{T}
  random::Bool
  order::Vector{Int}                     # buffer for random orders
  candidates::Union{Nothing, Vector{Int}}  # the coordinates of full passes, nothing for all
  passes::Int
  converged::Bool
  maxchange::T
  tolerance::T
end

_CDState(::Type{T}, x, random::Bool) where {T} =
  _CDState{T}(random, Vector{Int}(undef, random ? ncoordinates(x) : 0), nothing, 0, false, T(NaN), T(NaN))

# a uniformly random order of 1:n in order[1:n] (Fisher–Yates), drawn from the
# global random number generator with the calls of CoordinateDescent.jl, so that
# a seeded run visits the coordinates in the same order
function _shuffle!(order::Vector{Int}, n::Int)
  @inbounds for i in 1:n
    order[i] = i
  end
  @inbounds for i in 1:n-1
    j = rand(i:n)
    order[i], order[j] = order[j], order[i]
  end
  order
end

# one pass over the candidate coordinates (full) or the active ones; returns the
# largest change of a coordinate, scaled if `scaled`
function _pass!(x, m, g, st::_CDState{T}, full::Bool, scaled::Bool) where {T}
  maxh = if full
    st.candidates === nothing ? _fullpass!(x, m, g, st, scaled) : _candidatepass!(x, m, g, st, st.candidates, scaled)
  else
    _activepass!(x, m, g, st, scaled)
  end
  dropzeros!(x)
  maxh
end

function _fullpass!(x, m, g, st::_CDState{T}, scaled::Bool) where {T}
  n = ncoordinates(x)
  maxh = zero(T)
  if st.random
    order = _shuffle!(st.order, n)
    for i in 1:n
      h = _visit!(m, g, x, @inbounds(order[i]), scaled)
      h > maxh && (maxh = h)
    end
  else
    for k in 1:n
      h = _visit!(m, g, x, k, scaled)
      h > maxh && (maxh = h)
    end
  end
  maxh
end

function _candidatepass!(x, m, g, st::_CDState{T}, cand::Vector{Int}, scaled::Bool) where {T}
  n = length(cand)
  maxh = zero(T)
  if st.random
    order = _shuffle!(st.order, n)
    for i in 1:n
      h = _visit!(m, g, x, @inbounds(cand[order[i]]), scaled)
      h > maxh && (maxh = h)
    end
  else
    for i in 1:n
      h = _visit!(m, g, x, @inbounds(cand[i]), scaled)
      h > maxh && (maxh = h)
    end
  end
  maxh
end

# the active coordinates do not change during the pass: a coordinate that
# becomes zero stays stored until dropzeros!
function _activepass!(x, m, g, st::_CDState{T}, scaled::Bool) where {T}
  n = nactive(x)
  act = activecoordinates(x)
  maxh = zero(T)
  if st.random
    order = _shuffle!(st.order, n)
    for i in 1:n
      h = _visit!(m, g, x, @inbounds(act[order[i]]), scaled)
      h > maxh && (maxh = h)
    end
  else
    for i in 1:n
      h = _visit!(m, g, x, @inbounds(act[i]), scaled)
      h > maxh && (maxh = h)
    end
  end
  maxh
end

# Alternates full passes with passes over the active set; stops after two
# consecutive passes below the tolerance, the first of which is a full pass, so
# that the active set is checked against all coordinates before stopping. The
# model m must be current for x.
function _cd!(x, m, g, st::_CDState{T}, s::CDSolver) where {T}
  sy = s.relative ? _responsescale(m, x) : one(T)
  scaled = s.relative && sy > 0
  tol = scaled ? T(s.tol) * sy : T(s.tol)
  st.passes, st.converged, st.maxchange, st.tolerance = 0, false, T(NaN), tol
  converged = true
  for pass in 1:s.maxpasses
    maxh = _pass!(x, m, g, st, converged, scaled)
    previous = converged
    st.passes, st.maxchange = pass, maxh
    converged = maxh < tol
    if previous && converged
      st.converged = true
      break
    end
  end
  x
end

_warn_not_converged(st::_CDState, s::CDSolver) =
  st.converged || @warn "coordinate descent did not converge in maxpasses = $(s.maxpasses) passes (largest change of a coordinate in the last pass $(round(st.maxchange; sigdigits = 3)) > tolerance $(round(st.tolerance; sigdigits = 3))); the solution may be inaccurate."

"""
    solve!(x, f, g, method::CDSolver)

Minimize `f(x) + g(x)` by coordinate descent, starting from `x`, a
`SparseIterate`, a `SymmetricSparseIterate`, or an `AtomIterate` with the groups
of a group penalty `g`; `x` is overwritten with the solution. Returns a
[`CDResults`](@ref). See [`CDSolver`](@ref).
"""
function solve!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}, f::AbstractFunction,
                g::AbstractFunction, method::CDSolver)
  m = cdmodel(f, x)
  solve!(x, m, g, method)
end

function solve!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}, m::CDModel{T},
                g::AbstractFunction, method::CDSolver) where {T}
  _check_penalty(g, x)
  _check_model(m, g)
  initialize!(m, x)
  st = _CDState(T, x, method.random)
  _cd!(x, m, g, st, method)
  _warn_not_converged(st, method)
  CDResults(x, _value(m, x) + g(x), st.passes, st.converged, st.maxchange, st.tolerance, 0)
end
