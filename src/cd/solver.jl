####################################
#
#  Coordinate descent: options, results and the passes
#
####################################

"""
    CDSolver(; maxpasses = 100_000, tol = 1e-7, order = :random, relative = true, anderson = 0,
               momentum = nothing)

Coordinate descent for `f(x) + g(x)`, where `g` is separable
(`isseparable(g)`: `NormL1`, `ElasticNet`, `MCP`, `SCAD`, `IndBox`, …) or a
group penalty such as `GroupNormL2` or `GroupNormLinf` on an `AtomIterate`
with its groups, and `f` is a loss whose structure [`solve!`](@ref)
recognizes: `LeastSquaresLoss`, a least-squares `GLMLoss`, `QuadraticFunction`,
[`KronQuadratic`](@ref), or one of these scaled, as in
`(1/n) * LeastSquaresLoss(y, X)`.

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

With `anderson = K > 0` and `order = :cyclic`, the passes over an unchanged
active set are accelerated by Anderson extrapolation (Bertrand and Massias,
*Anderson acceleration of coordinate descent*, AISTATS 2021): every `K + 1`
such passes, the iterates on the active set are extrapolated, and the
extrapolated point is kept if it decreases `f + g`. On the lasso designs of the
benchmark, `CDSolver(; order = :cyclic, anderson = 5)` takes 1.6 to 2.6 times
less time than cyclic order alone at 0.02 λmax. A random order changes the map
of a pass from one pass to the next, which the extrapolation needs to repeat,
so `anderson` is ignored with it, as on an `AtomIterate`. Strongly correlated
columns still favor a random order.

With `momentum`, each pass over the active set starts from the extrapolated
point `xₖ + βₖ (xₖ - xₖ₋₁)`, with Nesterov's `βₖ = (k - 1) / (k + 2)`, as in
CoordinateDescent.jl 0.6. The momentum restarts after a full pass, when the
active set shrinks, and when the step of a pass points against it (O'Donoghue
and Candès, *Adaptive restart for accelerated gradient schemes*, 2015). Full
passes are never extrapolated, so the method stops on a plain full pass. The
cache of a built-in model moves with the
point, at the cost of a few vector operations per pass; a model of another
loss is initialized at the extrapolated point. By default
(`momentum = nothing`) it is used with the random order on a `SparseIterate`
for `LeastSquaresLoss`, `GLMLoss`, `SqrtLeastSquaresLoss`,
`QuadraticFunction` and [`SqrtQuadraticFunction`](@ref), and not otherwise. `momentum = true` asks for it with any
model on a `SparseIterate` or a `SymmetricSparseIterate`, such as
[`KronQuadratic`](@ref); `momentum = false` turns it off. With the cyclic order
it can stall far from the minimizer, so `order = :cyclic` refuses
`momentum = true`.
"""
struct CDSolver <: ProximalSolver
  maxpasses::Int
  tol::Float64
  random::Bool
  relative::Bool
  anderson::Int
  momentum::Union{Nothing, Bool}
end

function CDSolver(; maxpasses::Integer = 100_000, tol::Real = 1e-7, order::Symbol = :random,
                  relative::Bool = true, anderson::Integer = 0, momentum::Union{Nothing, Bool} = nothing)
  maxpasses >= 0 || throw(ArgumentError("maxpasses must be nonnegative, but it is $maxpasses."))
  tol >= 0 || throw(ArgumentError("tol must be nonnegative, but it is $tol."))
  order in (:random, :cyclic) || throw(ArgumentError("order must be :random or :cyclic, but it is :$order."))
  anderson >= 0 || throw(ArgumentError("anderson must be nonnegative, but it is $anderson."))
  momentum === true && order === :cyclic &&
    throw(ArgumentError("momentum needs order = :random; with the cyclic order it can stall far from the minimizer."))
  CDSolver(Int(maxpasses), Float64(tol), order === :random, relative, Int(anderson), momentum)
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

# Nesterov momentum on the passes over the active set (see CDSolver), as in
# CoordinateDescent.jl 0.6. Before a pass over the active set, the active values
# move to y = xₖ + βₖ (xₖ - xₖ₋₁), βₖ = (k - 1) / (k + 2), and so does an affine
# cache of the model (_cache), exactly up to rounding; another model is
# initialized at y. After the pass, k restarts at 1 when the step of the pass
# points against the momentum, (xₖ₊₁ - y)'(xₖ - xₖ₋₁) < 0. The history is
# indexed by position among the stored values, x.nzval[1:m], so it starts again
# when the active set changes: after every full pass, and when dropzeros! has
# compacted the stored values. The buffers have room for every coordinate, so
# that no pass allocates.
mutable struct _Momentum{T}
  xprev::Vector{T}      # the active values after the pass before the last
  xcur::Vector{T}       # the active values after the last pass
  y::Vector{T}          # the active values the last pass started from
  cprev::Vector{T}      # the cache of the model at xprev
  ccur::Vector{T}       # the cache of the model at xcur
  n::Int                # number of active values in the history
  k::Int
  moved::Bool           # whether the cache moved since it was last computed
end

_Momentum{T}() where {T} = _Momentum{T}(T[], T[], T[], T[], T[], 0, 1, false)

# the state of one run: the coordinate order, the momentum, and what the passes
# report
mutable struct _CDState{T}
  random::Bool
  order::Vector{Int}                     # buffer for random orders
  candidates::Union{Nothing, Vector{Int}}  # the coordinates of full passes, nothing for all
  momentum::_Momentum{T}
  passes::Int
  converged::Bool
  maxchange::T
  tolerance::T
end

_CDState(::Type{T}, x, random::Bool) where {T} =
  _CDState{T}(random, Vector{Int}(undef, random ? ncoordinates(x) : 0), nothing, _Momentum{T}(), 0, false,
              T(NaN), T(NaN))

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
  s.anderson > 0 && !s.random && !(x isa AtomIterate) && return _cd_anderson!(x, m, g, st, s)
  _usemomentum(s, x, m) && return _cd_momentum!(x, m, g, st, s)
  sy = s.relative ? response_scale(m, x) : one(T)
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

# whether the passes use momentum; see CDSolver
function _usemomentum(s::CDSolver, x, m)
  s.momentum === false && return false
  s.momentum === nothing && return s.random && x isa SparseIterate && _momentum_default(m)
  x isa Union{SparseIterate, SymmetricSparseIterate} ||
    throw(ArgumentError("momentum needs a SparseIterate or a SymmetricSparseIterate, not a $(typeof(x))."))
  true
end

# the cache of a model that is affine in x, which momentum moves with the
# point; nothing for a model that momentum initializes at the point instead
_cache(m::CDModel) = nothing

# called after momentum moved the cache with the point, for a model that derives
# from the cache what is not affine in x (as SqrtQuadModel its value)
_cache_moved!(m::CDModel, x) = m

# whether momentum is used by default, with a random order on a SparseIterate:
# for the models whose cache moves with the point, where it pays off
_momentum_default(m::CDModel) = _cache(m) !== nothing

# the same passes with momentum on the passes over the active set; the cache
# moved by the momentum is computed again before the next full pass, so that it
# does not drift into the stopping rule or the optimality checks of screening
function _cd_momentum!(x, m, g, st::_CDState{T}, s::CDSolver) where {T}
  sy = s.relative ? response_scale(m, x) : one(T)
  scaled = s.relative && sy > 0
  tol = scaled ? T(s.tol) * sy : T(s.tol)
  st.passes, st.converged, st.maxchange, st.tolerance = 0, false, T(NaN), tol
  mom = _momentum_start!(st.momentum, x, m)
  converged = true
  for pass in 1:s.maxpasses
    full = converged
    full || _momentum_step!(mom, x, m)
    maxh = _pass!(x, m, g, st, full, scaled)
    previous = converged
    st.passes, st.maxchange = pass, maxh
    converged = maxh < tol
    if previous && converged
      st.converged = true
      break
    end
    if full
      _momentum_reset!(mom, x, m)
    elseif converged
      _momentum_refresh!(mom, x, m)
    else
      _momentum_update!(mom, x, m)
    end
  end
  x
end

# a new history for a run, with buffers for every coordinate and for the cache
function _momentum_start!(s::_Momentum, x, m)
  n = ncoordinates(x)
  if length(s.xcur) < n
    resize!(s.xprev, n); resize!(s.xcur, n); resize!(s.y, n)
  end
  c = _cache(m)
  if c !== nothing && length(s.ccur) != length(c)
    resize!(s.ccur, length(c)); resize!(s.cprev, length(c))
  end
  s.n, s.k, s.moved = 0, 1, false
  s
end

# starts the history at x, as after a full pass or when the active set shrank
function _momentum_reset!(s::_Momentum, x, m)
  n = nactive(x)
  s.n, s.k = n, 1
  copyto!(s.xcur, 1, x.nzval, 1, n)
  copyto!(s.xprev, 1, s.xcur, 1, n)
  c = _cache(m)
  if c !== nothing
    copyto!(s.ccur, c)
    copyto!(s.cprev, c)
  end
  s
end

# moves x and the cache of m to the point the next pass over the active set
# starts from
function _momentum_step!(s::_Momentum{T}, x, m) where {T}
  n = s.n
  β = s.k > 1 ? T((s.k - 1) / (s.k + 2)) : zero(T)
  @inbounds for j in 1:n
    s.y[j] = s.xcur[j] + β * (s.xcur[j] - s.xprev[j])
  end
  iszero(β) && return s
  copyto!(x.nzval, 1, s.y, 1, n)
  c = _cache(m)
  if c === nothing
    initialize!(m, x)
  else
    @inbounds @simd for i in eachindex(c)
      c[i] = s.ccur[i] + β * (s.ccur[i] - s.cprev[i])
    end
    _cache_moved!(m, x)
    s.moved = true
  end
  s
end

# after a pass over the active set: restarts when its step points against the
# momentum, and records the new point
function _momentum_update!(s::_Momentum{T}, x, m) where {T}
  nactive(x) == s.n || return _momentum_reset!(s, x, m)
  n = s.n
  d = zero(T)
  @inbounds for j in 1:n
    d += (x.nzval[j] - s.y[j]) * (s.xcur[j] - s.xprev[j])
  end
  s.k = s.k > 1 && d < 0 ? 1 : s.k + 1
  s.xprev, s.xcur = s.xcur, s.xprev
  copyto!(s.xcur, 1, x.nzval, 1, n)
  c = _cache(m)
  if c !== nothing
    s.cprev, s.ccur = s.ccur, s.cprev
    copyto!(s.ccur, c)
  end
  s
end

# computes the cache again where repeated moves may have let it drift
function _momentum_refresh!(s::_Momentum, x, m)
  s.moved && initialize!(m, x)
  s.moved = false
  s
end

# the same passes with Anderson extrapolation on the active set
function _cd_anderson!(x, m, g, st::_CDState{T}, s::CDSolver) where {T}
  sy = s.relative ? response_scale(m, x) : one(T)
  scaled = s.relative && sy > 0
  tol = scaled ? T(s.tol) * sy : T(s.tol)
  st.passes, st.converged, st.maxchange, st.tolerance = 0, false, T(NaN), tol
  acc = _Anderson(T, s.anderson)
  converged = true
  for pass in 1:s.maxpasses
    full = converged
    maxh = _pass!(x, m, g, st, full, scaled)
    previous = converged
    st.passes, st.maxchange = pass, maxh
    converged = maxh < tol
    if previous && converged
      st.converged = true
      break
    end
    _extrapolate!(acc, x, m, g, full)
  end
  x
end

# the iterates on a fixed active set S after each pass, K + 1 of them
mutable struct _Anderson{T}
  K::Int
  S::Vector{Int}
  hist::Matrix{T}
  nh::Int
  saved::Vector{T}
end

_Anderson(::Type{T}, K::Int) where {T} = _Anderson{T}(K, Int[], Matrix{T}(undef, 0, K + 1), 0, T[])

# Record x on the active set; with K + 1 records, replace x by the combination
# of the last K records with weights c ∝ (U'U)⁻¹1, U the differences of the
# records, if that decreases f + g. A full pass, or a change of the active set,
# starts a new history.
function _extrapolate!(a::_Anderson{T}, x, m, g, full::Bool) where {T}
  act = activecoordinates(x)
  if full || length(act) != length(a.S) || any(i -> @inbounds(act[i] != a.S[i]), eachindex(a.S))
    copyto!(resize!(a.S, length(act)), act)
    size(a.hist, 1) == length(a.S) || (a.hist = Matrix{T}(undef, length(a.S), a.K + 1))
    a.nh = 0
  end
  isempty(a.S) && return a
  cx = coordinates(x)
  a.nh += 1
  for (i, k) in enumerate(a.S)
    @inbounds a.hist[i, a.nh] = cx[k]
  end
  a.nh == a.K + 1 || return a
  a.nh = 0
  U = diff(a.hist; dims = 2)
  M = U' * U
  M += (T(1e-10) * max(tr(M), floatmin(T))) * I
  c = M \ ones(T, a.K)
  sc = sum(c)
  (isfinite(sc) && !iszero(sc)) || return a
  c ./= sc
  xe = view(a.hist, :, 2:a.K+1) * c
  F = loss_value(m, x) + g(x)
  resize!(a.saved, length(a.S))
  for (i, k) in enumerate(a.S)
    @inbounds a.saved[i] = cx[k]
    cx[k] = xe[i]
  end
  initialize!(m, x)
  if !(loss_value(m, x) + g(x) < F)
    for (i, k) in enumerate(a.S)
      cx[k] = a.saved[i]
    end
    initialize!(m, x)
  end
  a
end

_warn_not_converged(st::_CDState, s::CDSolver) =
  st.converged || @warn "coordinate descent did not converge in maxpasses = $(s.maxpasses) passes (largest change of a coordinate in the last pass $(round(st.maxchange; sigdigits = 3)) > tolerance $(round(st.tolerance; sigdigits = 3))); the solution may be inaccurate."

"""
    solve!(x, f, g, method::CDSolver)

Minimize `f(x) + g(x)` by coordinate descent, starting from `x`, a
`SparseIterate`, a `SymmetricSparseIterate`, or an `AtomIterate` with the groups
of a group penalty `g`; `x` is overwritten with the solution. A dense vector `x`
is converted to a `SparseIterate` (an `AtomIterate` for a group penalty) and
overwritten with the solution. Returns a [`CDResults`](@ref). See
[`CDSolver`](@ref).
"""
function solve!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}, f::AbstractFunction,
                g::AbstractFunction, method::CDSolver)
  m = cdmodel(f, x)
  solve!(x, m, g, method)
end

# a dense vector is converted to an iterate and overwritten with the solution:
# an AtomIterate with the groups of a group penalty, a SparseIterate otherwise
solve!(x::Vector{T}, f::AbstractFunction, g::AbstractFunction, method::CDSolver; kwargs...) where {T<:AbstractFloat} =
  _solve_dense!(x, f, g, method; kwargs...)
solve!(x::Vector{T}, m::CDModel, g::AbstractFunction, method::CDSolver; kwargs...) where {T<:AbstractFloat} =
  _solve_dense!(x, m, g, method; kwargs...)

function _solve_dense!(x::Vector, f, g, method::CDSolver; kwargs...)
  xs = g isa ProximalBase.AbstractGroupNorm ? AtomIterate(x, g.groups) : SparseIterate(x)
  r = solve!(xs, f, g, method; kwargs...)
  copyto!(x, Vector(xs))
  CDResults(x, r.minimum, r.passes, r.converged, r.maxchange, r.tolerance, r.kkt_checks)
end

function solve!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}, m::CDModel{T},
                g::AbstractFunction, method::CDSolver) where {T}
  _check_iterate(g, x)
  check_penalty(m, g)
  initialize!(m, x)
  st = _CDState(T, x, method.random)
  _cd!(x, m, g, st, method)
  _warn_not_converged(st, method)
  CDResults(x, loss_value(m, x) + g(x), st.passes, st.converged, st.maxchange, st.tolerance, 0)
end
