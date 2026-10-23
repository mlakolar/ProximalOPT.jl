####################################
#
#  least squares  s ∑ᵢ wᵢ (yᵢ - oᵢ - Xᵢ'β)² / 2
#
####################################

# The model of LeastSquaresLoss (s = 1, w = 1, o = 0), of a GLMLoss with the
# Squared loss (weights w, offset o), and of either scaled by s. It caches the
# residuals r = y - o - Xβ, so a coordinate step costs O(n), and the weighted
# squared column norms, computed on the first visit to each coordinate after
# initialize!. With an intercept, the loss is minimized over an unpenalized
# intercept β₀ as well, whose minimizer is β₀ = ȳ - x̄'β for the column means x̄
# and the mean ȳ of y - o; the columns and y are centered on the fly, so X is
# not copied.
mutable struct LSModel{T<:AbstractFloat, MX<:AbstractMatrix{T}, VY<:AbstractVector{T},
                       W<:Union{Nothing, AbstractVector{T}}, O<:Union{Nothing, AbstractVector{T}}} <: CDModel{T}
  X::MX
  y::VY
  w::W
  offset::O
  s::T
  r::Vector{T}
  a::Vector{T}          # ∑ᵢ wᵢ (Xᵢₖ - x̄ₖ)², NaN until computed
  xbar::Vector{T}       # column means with an intercept, otherwise zeros
  ybar::T               # mean of y - o with an intercept, otherwise zero
  L::Vector{T}          # curvatures of the groups of an AtomIterate, NaN until computed
  tmp::Vector{T}        # wᵢ rᵢ, for the gradient of a weighted loss
end

function LSModel(y::AbstractVector, X::AbstractMatrix, w, offset, s::Real; intercept::Bool = false)
  T = float(promote_type(eltype(y), eltype(X)))
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  (intercept && w !== nothing) && throw(ArgumentError("an intercept is not supported with weights"))
  Xt, yt = convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y)
  wt = w === nothing ? nothing : convert(AbstractVector{T}, w)
  ot = offset === nothing ? nothing : convert(AbstractVector{T}, offset)
  _check_finite(yt, "y"); _check_finite(Xt, "X")
  xbar = intercept ? _column_means(Xt) : zeros(T, p)
  ybar = intercept ? _mean_response(yt, ot) : zero(T)
  LSModel{T, typeof(Xt), typeof(yt), typeof(wt), typeof(ot)}(Xt, yt, wt, ot, T(s), zeros(T, n), fill(T(NaN), p),
                                                             xbar, ybar, T[], wt === nothing ? T[] : zeros(T, n))
end

_mean_response(y, ::Nothing) = sum(y) / length(y)
_mean_response(y, o) = (sum(y) - sum(o)) / length(y)

cdmodel(f::LeastSquaresLoss{T, <:AbstractVector}, x; intercept::Bool = false) where {T} =
  LSModel(f.Y, f.X, nothing, nothing, one(T); intercept)
cdmodel(f::Scaled{<:LeastSquaresLoss{T, <:AbstractVector}}, x; intercept::Bool = false) where {T} =
  LSModel(f.f.Y, f.f.X, nothing, nothing, f.α; intercept)
cdmodel(f::GLMLoss{ProximalBase.Squared}, x; intercept::Bool = false) =
  LSModel(f.y, f.X, f.weights, f.offset, 1; intercept)
cdmodel(f::Scaled{<:GLMLoss{ProximalBase.Squared}}, x; intercept::Bool = false) =
  LSModel(f.f.y, f.f.X, f.f.weights, f.f.offset, f.α; intercept)

"""
    intercept(m::CDModel, x)

The unpenalized intercept that goes with `x` in a least-squares model built
with `intercept = true` (see [`cdmodel`](@ref)); zero without one.
"""
intercept(m::LSModel, x::AbstractArray) = m.ybar - dot(m.xbar, x)

"""
    residuals(m::CDModel)

The residuals `y - o - β₀ - Xβ` of a least-squares model, for the iterate it is
current for.
"""
residuals(m::LSModel) = m.r

_check_finite(A::AbstractArray, name::AbstractString) =
  all(isfinite, A) || throw(ArgumentError("$name contains NaN or Inf."))

# the column means of X; a constant column gets its value exactly, so that it is
# exactly zero once centered (the computed mean can differ in the last bits)
function _column_means(X::AbstractMatrix{T}) where {T}
  n = size(X, 1)
  xbar = Vector{T}(vec(sum(X; dims = 1)) ./ n)
  n > 0 || return xbar
  for k in axes(X, 2)
    c = X[1, k]
    all(==(c), view(X, :, k)) && (xbar[k] = c)
  end
  xbar
end

# the entries of the active coordinates of x: the stored entries of a
# SparseIterate, the entries of the active groups of an AtomIterate
_activeentries(x::SparseIterate) = activecoordinates(x)
_activeentries(x::AtomIterate) = Iterators.flatten(x.groups[k] for k in activecoordinates(x))

function initialize!(m::LSModel{T}, x::Union{SparseIterate{T}, AtomIterate{T}}) where {T}
  length(x) == size(m.X, 2) || throw(DimensionMismatch("x has length $(length(x)) and X has $(size(m.X, 2)) columns"))
  r, y, X = m.r, m.y, m.X
  if m.offset === nothing
    r .= y .- m.ybar
  else
    r .= y .- m.offset .- m.ybar
  end
  for k in _activeentries(x)
    v = x[k]
    iszero(v) && continue
    _column_axpy!(r, X, k, m.xbar[k], v)
  end
  fill!(m.a, T(NaN))          # X may have changed in place
  x isa AtomIterate && fill!(resize!(m.L, ncoordinates(x)), T(NaN))
  m
end

# Columns centered at c, where c is the column mean when the model fits an
# intercept and zero otherwise; a zero c takes the plain loops.

# (a, b) with b = (X[:, k] - c)' r and a = ‖X[:, k] - c‖², which is computed
# only when the cached a is NaN
@inline function _column_dot(X::AbstractMatrix{T}, r::AbstractVector{T}, k::Int, c::T, a::T) where {T}
  b = zero(T)
  if isnan(a)
    a = zero(T)
    if iszero(c)
      @inbounds @simd for i in eachindex(r)
        a += X[i, k] * X[i, k]
        b += r[i] * X[i, k]
      end
    else
      @inbounds @simd for i in eachindex(r)
        d = X[i, k] - c
        a += d * d
        b += r[i] * d
      end
    end
  elseif iszero(c)
    @inbounds @simd for i in eachindex(r)
      b += r[i] * X[i, k]
    end
  else
    @inbounds @simd for i in eachindex(r)
      b += r[i] * (X[i, k] - c)
    end
  end
  a, b
end

# the same with weights: b = ∑ᵢ wᵢ Xᵢₖ rᵢ and a = ∑ᵢ wᵢ Xᵢₖ²
@inline function _column_dot(X::AbstractMatrix{T}, r::AbstractVector{T}, w::AbstractVector{T}, k::Int, a::T) where {T}
  b = zero(T)
  if isnan(a)
    a = zero(T)
    @inbounds @simd for i in eachindex(r)
      a += X[i, k] * X[i, k] * w[i]
      b += r[i] * X[i, k] * w[i]
    end
  else
    @inbounds @simd for i in eachindex(r)
      b += r[i] * X[i, k] * w[i]
    end
  end
  a, b
end

# r -= h (X[:, k] - c)
@inline function _column_axpy!(r::AbstractVector{T}, X::AbstractMatrix{T}, k::Int, c::T, h::T) where {T}
  if iszero(c)
    @inbounds @simd for i in eachindex(r)
      r[i] -= X[i, k] * h
    end
  else
    @inbounds @simd for i in eachindex(r)
      r[i] -= (X[i, k] - c) * h
    end
  end
  r
end

@inline _column_dot(m::LSModel, k::Int) = _column_dot(m, m.w, k)
@inline _column_dot(m::LSModel, ::Nothing, k::Int) = _column_dot(m.X, m.r, k, m.xbar[k], m.a[k])
@inline _column_dot(m::LSModel, w::AbstractVector, k::Int) = _column_dot(m.X, m.r, w, k, m.a[k])

# ∑ᵢ wᵢ (Xᵢₖ - x̄ₖ)², from the cache, computed and cached if needed
function _sqnorm!(m::LSModel{T}, k::Int) where {T}
  a = m.a[k]
  if isnan(a)
    X, c = m.X, m.xbar[k]
    a = zero(T)
    if m.w === nothing
      @inbounds @simd for i in axes(X, 1)
        a += (X[i, k] - c) * (X[i, k] - c)
      end
    else
      w = m.w
      @inbounds @simd for i in axes(X, 1)
        a += X[i, k] * X[i, k] * w[i]
      end
    end
    m.a[k] = a
  end
  a
end

# the coordinate goes to xₖ + b/a, with step 1/(s a) for the penalty; a column
# that the loss does not depend on (a zero column, or a constant one with an
# intercept) gets the minimizer of the penalty nearest zero
@inline function coordinate_step!(m::LSModel{T}, g, x::SparseIterate{T}, k::Int) where {T}
  a, b = _column_dot(m, k)
  m.a[k] = a
  old = x[k]
  new = iszero(a) ? prox_entry(g, zero(T), k, floatmax(T)) : prox_entry(g, old + b / a, k, inv(m.s * a))
  h = new - old
  iszero(h) && return h      # x and the residuals do not change
  x[k] = new
  _column_axpy!(m.r, m.X, k, m.xbar[k], h)
  h
end

function coordinate_derivatives(m::LSModel{T}, x::SparseIterate{T}, k::Int) where {T}
  a, b = _column_dot(m, k)
  m.a[k] = a
  -m.s * b, m.s * a
end

update_cache!(m::LSModel{T}, x::SparseIterate{T}, k::Int, h::T) where {T} = (_column_axpy!(m.r, m.X, k, m.xbar[k], h); m)

# the residuals are affine in β, also with the centering of an intercept
_cache(m::LSModel) = m.r

coordinate_scale(m::LSModel, x, k::Int) = sqrt(m.s * _sqnorm!(m, k))

function response_scale(m::LSModel{T}, x) where {T}
  y, o, w, ȳ = m.y, m.offset, m.w, m.ybar
  s = zero(T)
  @inbounds for i in eachindex(y)
    d = (o === nothing ? y[i] : y[i] - o[i]) - ȳ
    s += (w === nothing ? d * d : w[i] * d * d)
  end
  sqrt(m.s * s)
end

function loss_value(m::LSModel{T}, x) where {T}
  r, w = m.r, m.w
  v = zero(T)
  if w === nothing
    @inbounds @simd for i in eachindex(r)
      v += r[i] * r[i]
    end
  else
    @inbounds @simd for i in eachindex(r)
      v += w[i] * r[i] * r[i]
    end
  end
  m.s * v / 2
end

# -s (X - 1x̄')' W r, with one product
function gradient!(G::AbstractVector{T}, m::LSModel{T}, x) where {T}
  if m.w === nothing
    mul!(G, transpose(m.X), m.r)
    sr = sum(m.r)
    G .= m.s .* (m.xbar .* sr .- G)
  else
    m.tmp .= m.w .* m.r
    mul!(G, transpose(m.X), m.tmp)
    G .*= -m.s
  end
  G
end

# the curvature of f over group k of x, s λmax((X[:, G] - 1x̄_G')' W (X[:, G] - 1x̄_G')),
# from the cache, computed and cached if needed
function _groupcurvature!(m::LSModel{T}, x::AtomIterate{T}, k::Int) where {T}
  L = m.L[k]
  if isnan(L)
    G = x.groups[k]
    if length(G) == 1
      L = m.s * _sqnorm!(m, first(G))
    else
      XG = m.X[:, G] .- transpose(m.xbar[G])
      m.w === nothing || (XG .*= sqrt.(m.w))
      L = max(m.s * eigmax(Symmetric(XG' * XG)), zero(T))
    end
    m.L[k] = L
  end
  L
end

# group k of x ← the proximal operator of λ0 wₖ ‖⋅‖₂ with step 1/L at
# x[G] - ∇_G f(x) / L, where L is the curvature of f over G, so that the
# quadratic with curvature L majorizes f over the group (groupwise majorization
# descent, Yang and Zou, 2015). A group at zero stays there if
# ‖∇_G f(x)‖ ≤ λ0 wₖ, whatever L, which is then not needed. Returns the largest
# change of an entry, scaled with `scaled`; u is a buffer of at least |G| entries.
function _groupstep!(m::LSModel{T}, g::GroupNormL2, x::AtomIterate{T}, k::Int, scaled::Bool) where {T}
  X, r = m.X, m.r
  G = x.groups[k]
  c = T(g.λ0 * g.w[k])
  u = _groupbuffer!(m, length(G))

  # u = -∇_G f(x) / s, where ∇ⱼf(x) = -s (X[:, j] - x̄ⱼ)' W r
  t = zero(T)
  for (i, j) in enumerate(G)
    a, b = _column_dot(m, j)
    m.a[j] = a
    u[i] = b
    t += b * b
  end
  !isactive(x, k) && m.s * sqrt(t) <= c && return zero(T)

  # u = x[G] - ∇_G f(x) / L
  L = _groupcurvature!(m, x, k)
  t = zero(T)
  for (i, j) in enumerate(G)
    v = iszero(L) ? x[j] : x[j] + m.s * u[i] / L
    u[i] = v
    t += v * v
  end

  shrink = _groupshrink(sqrt(t), c, L)
  maxh = zero(T)
  for (i, j) in enumerate(G)
    new = shrink * u[i]
    h = new - x[j]
    iszero(h) && continue
    x[j] = new
    _column_axpy!(r, X, j, m.xbar[j], h)
    maxh = max(maxh, _scaledchange(m, x, j, h, scaled))
  end
  maxh
end

# a buffer of n entries for group steps, kept in the model
_groupbuffer!(m::LSModel{T}, n::Int) where {T} = (length(m.tmp) < n && resize!(m.tmp, n); m.tmp)
