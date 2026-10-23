####################################
#
#  The losses of coordinate descent, and what it needs from iterates and penalties
#
####################################

"""
    CDModel{T}

A loss together with the cache that makes its coordinate updates cheap, such as
the residuals of least squares. [`cdmodel`](@ref) builds one from a loss of
ProximalBase; [`solve!`](@ref) accepts a model in place of the loss, so that a
model built once serves many solves (paths, reweighting).

A model `m` implements, for the iterates it supports:

- `initialize!(m, x)`: make the cache current for `x`;
- `ProximalOPT._coordinate(m, x, k)`: `(G, a)`, the derivative of the loss
  along coordinate `k` of `x` and its curvature `a` (exact for a quadratic,
  otherwise a bound), from the cache;
- `ProximalOPT._update!(m, x, k, h)`: update the cache for the change `h` of
  coordinate `k`, which the caller has already written into `x`;
- `ProximalOPT._coordscale(m, x, k)` and `ProximalOPT._responsescale(m, x)`, the
  scales of the relative stopping rule (see [`CDSolver`](@ref));
- `ProximalOPT._value(m, x)`, the value of the loss at `x`, from the cache;
- `gradient!(G, m, x)`, the gradient at `x`, from the cache, as an array of the
  size of `x` (one entry per entry of `x`, also for a `SymmetricSparseIterate`).

From these, a coordinate step goes to [`prox_entry`](@ref)`(g, xₖ - G/a, i, m/a)`,
where `i` is the index of the entry in `x` and `m` the number of entries the
coordinate moves (2 for an off-diagonal coordinate of a
`SymmetricSparseIterate`): exact for quadratic losses, and a majorization step
when `a` bounds the curvature. A model may instead define the whole step,
`ProximalOPT._step!(m, g, x, k)`, which returns the change of the coordinate;
the built-in models do, to fuse the computations.
"""
abstract type CDModel{T} end

"""
    initialize!(m::CDModel, x)

Make the cache of the model `m` current for the iterate `x`.
"""
function initialize! end

"""
    cdmodel(f, x)

The [`CDModel`](@ref) of the loss `f` for iterates like `x`: for
`LeastSquaresLoss`, a least-squares `GLMLoss` (with weights and an offset),
`QuadraticFunction` and [`KronQuadratic`](@ref), each possibly scaled, as in
`(1/n) * LeastSquaresLoss(y, X)`.
"""
function cdmodel end

cdmodel(f::AbstractFunction, x) =
  throw(ArgumentError("coordinate descent does not know the structure of $(typeof(f)); see the documentation of cdmodel."))

# the coordinate step from _coordinate and _update!, for models that do not
# define their own; a coordinate with zero curvature, which the loss does not
# depend on, goes to the minimizer of the penalty nearest zero
@inline function _step!(m::CDModel{T}, g, x::SparseIterate{T}, k::Int) where {T}
  G, a = _coordinate(m, x, k)
  old = x[k]
  new = iszero(a) ? prox_entry(g, zero(T), k, floatmax(T)) : prox_entry(g, old - G / a, k, inv(a))
  h = new - old
  iszero(h) && return h
  x[k] = new
  _update!(m, x, k, h)
  h
end

@inline function _step!(m::CDModel{T}, g, x::SymmetricSparseIterate{T}, k::Int) where {T}
  G, a = _coordinate(m, x, k)
  r, c = tril_cartesian(x.p, k)
  cx = coordinates(x)
  old = cx[k]
  i = _entry(x.p, r, c)
  new = iszero(a) ? prox_entry(g, zero(T), i, floatmax(T)) : prox_entry(g, old - G / a, i, _multiplicity(T, r, c) / a)
  h = new - old
  iszero(h) && return h
  cx[k] = new
  _update!(m, x, k, h)
  h
end

# the change of coordinate k in the stopping rule; for the groups of an
# AtomIterate, _groupstep! returns the largest scaled change of an entry
@inline function _visit!(m::CDModel, g, x, k::Int, scaled::Bool)
  h = abs(_step!(m, g, x, k))
  scaled ? h * _coordscale(m, x, k) : h
end

@inline _visit!(m::CDModel, g, x::AtomIterate, k::Int, scaled::Bool) = _groupstep!(m, g, x, k, scaled)

####################################
#  coordinates of the iterates
####################################

# For a SymmetricSparseIterate, coordinate k is the lower-triangle entry (r, c),
# r ≥ c: its index as an entry of the p × p matrix, which penalties take, and
# the number of entries it sets, which multiplies the penalty
@inline _entry(p::Int, r::Int, c::Int) = (c - 1) * p + r
@inline _multiplicity(::Type{T}, r::Int, c::Int) where {T} = r == c ? one(T) : T(2)

# the size of the gradient over coordinate k, and the weight of the penalty on
# it, for screening and for λmax
_gradnorm(grad::AbstractArray, ::SparseIterate, k::Int) = abs(grad[k])
function _gradnorm(grad::AbstractArray, x::SymmetricSparseIterate, k::Int)
  r, c = tril_cartesian(x.p, k)
  abs(grad[_entry(x.p, r, c)])
end
_gradnorm(grad::AbstractArray, x::AtomIterate, k::Int) = norm(view(grad, x.groups[k]))

_coordweight(g, ::SparseIterate, k::Int) = _weight(g, k)
function _coordweight(g, x::SymmetricSparseIterate, k::Int)
  r, c = tril_cartesian(x.p, k)
  _weight(g, _entry(x.p, r, c))
end
_coordweight(g::GroupNormL2, ::AtomIterate, k::Int) = g.w[k]

_isnonzero(x::SparseIterate, k::Int) = !iszero(x[k])
_isnonzero(x::SymmetricSparseIterate, k::Int) = !iszero(coordinates(x)[k])
_isnonzero(x::AtomIterate, k::Int) = isactive(x, k) && any(!iszero, atom(x, k))

_gradient_buffer(x::AbstractArray{T}) where {T} = similar(Array{T}, axes(x))

####################################
#  penalties
####################################

# The penalty level λ of g, g with level λ instead, and the weight of entry i
# (of group i for a group penalty): a coordinate at zero is optimal while the
# gradient over it is at most λ times its weight. Paths and screening need these.
_level(g::Union{NormL1, NormL1Nonneg, GroupNormL2}) = g.λ0
_level(g::ElasticNet) = g.λ1
_level(g::Union{MCP, SCAD}) = g.λ

_with_level(g::NormL1, λ) = NormL1(λ, g.w)
_with_level(g::NormL1Nonneg, λ) = NormL1Nonneg(λ, g.w)
_with_level(g::GroupNormL2, λ) = GroupNormL2(λ, g.groups, g.w)
_with_level(g::ElasticNet, λ) = ElasticNet(λ, g.λ2, g.w)
_with_level(g::MCP, λ) = MCP(λ, g.a)
_with_level(g::SCAD, λ) = SCAD(λ, g.a)
_with_level(g, λ) = throw(ArgumentError("$(typeof(g)) has no penalty level for a path; use NormL1, ElasticNet, MCP, SCAD or GroupNormL2."))

_weight(g::Union{NormL1, NormL1Nonneg}, i::Int) = _entryweight(g.w, i, typeof(g.λ0))
_weight(g::ElasticNet, i::Int) = _entryweight(g.w, i, typeof(g.λ1))
_entryweight(::Nothing, ::Int, ::Type{T}) where {T} = one(T)
_entryweight(w::AbstractArray, i::Int, ::Type) = @inbounds w[i]
_weight(g::Union{MCP{T}, SCAD{T}}, ::Int) where {T} = one(T)
_weight(g::GroupNormL2, k::Int) = g.w[k]
_weight(g, ::Int) = throw(ArgumentError("$(typeof(g)) has no penalty level for paths and screening; use NormL1, ElasticNet, MCP, SCAD or GroupNormL2."))

_haslevel(g) = applicable(_level, g)

# a model that takes only some penalties says so here
_check_model(m::CDModel, g) = nothing

function _check_penalty(g::AbstractFunction, x)
  isseparable(g) && return nothing
  if g isa GroupNormL2
    x isa AtomIterate || throw(ArgumentError("GroupNormL2 needs an AtomIterate with its groups, as in AtomIterate(p, g.groups)."))
    x.groups === g.groups || x.groups == g.groups ||
      throw(ArgumentError("The groups of x must be the groups of g, as in AtomIterate(p, g.groups)."))
    return nothing
  end
  throw(ArgumentError("coordinate descent needs a separable penalty (isseparable(g)) or GroupNormL2, but g is a $(typeof(g))."))
end

# the factor max(1 - c / (L ‖u‖), 0) by which the proximal operator of c‖⋅‖₂
# with step 1/L shrinks a vector u, given nu = ‖u‖. With L = 0 the loss does not
# depend on the group, which then goes to zero if it is penalized and keeps its
# values otherwise
function _groupshrink(nu::T, c::T, L::T) where {T<:AbstractFloat}
  iszero(L) && return iszero(c) ? one(T) : zero(T)
  nu > c / L ? one(T) - c / (L * nu) : zero(T)
end

# the change of an entry in the stopping rule, scaled as in _visit!
_scaledchange(m, x, j::Int, h, scaled::Bool) = scaled ? abs(h) * _coordscale(m, x, j) : abs(h)
