####################################
#
#  square-root least squares  ‖y - Xβ‖
#
####################################

"""
    SqrtLeastSquaresLoss(y, X)

The loss `f(β) = ‖y - Xβ‖₂` of the square-root lasso (Belloni, Chernozhukov and
Wang, 2011), whose optimal penalty level does not depend on the noise level.
It is not differentiable where the residuals vanish, so coordinate descent
([`CDSolver`](@ref)) minimizes `f + g` exactly over each coordinate, which it
does for `g = NormL1(λ, w)`. With [`cdmodel`](@ref)`(f, x; intercept = true)`
the loss is minimized over an unpenalized intercept as well.
"""
struct SqrtLeastSquaresLoss{T<:AbstractFloat, Ty<:AbstractVector, TX<:AbstractMatrix} <: AbstractFunction
  y::Ty
  X::TX
  tmp::Vector{T}
end

function SqrtLeastSquaresLoss(y::AbstractVector, X::AbstractMatrix)
  size(X, 1) == length(y) || throw(DimensionMismatch("X has $(size(X, 1)) rows and y has length $(length(y))"))
  T = float(promote_type(eltype(y), eltype(X)))
  SqrtLeastSquaresLoss{T, typeof(y), typeof(X)}(y, X, zeros(T, length(y)))
end

ProximalBase.isconvex(::Type{<:SqrtLeastSquaresLoss}) = true

function (f::SqrtLeastSquaresLoss)(β::AbstractVector)
  mul!(f.tmp, f.X, β)
  f.tmp .= f.y .- f.tmp
  norm(f.tmp)
end

mutable struct SqrtLSModel{T<:AbstractFloat, MX<:AbstractMatrix{T}, VY<:AbstractVector{T}} <: CDModel{T}
  X::MX
  y::VY
  r::Vector{T}
  a::Vector{T}          # squared norms of the centered columns, NaN until computed
  xbar::Vector{T}
  ybar::T
end

function cdmodel(f::SqrtLeastSquaresLoss{T}, x; intercept::Bool = false) where {T}
  X, y = convert(AbstractMatrix{T}, f.X), convert(AbstractVector{T}, f.y)
  _check_finite(y, "y"); _check_finite(X, "X")
  p = size(X, 2)
  SqrtLSModel{T, typeof(X), typeof(y)}(X, y, zeros(T, length(y)), fill(T(NaN), p),
                                       intercept ? _column_means(X) : zeros(T, p),
                                       intercept ? _mean_response(y, nothing) : zero(T))
end

intercept(m::SqrtLSModel, x::AbstractArray) = m.ybar - dot(m.xbar, x)
residuals(m::SqrtLSModel) = m.r

function initialize!(m::SqrtLSModel{T}, x::SparseIterate{T}) where {T}
  length(x) == size(m.X, 2) || throw(DimensionMismatch("x has length $(length(x)) and X has $(size(m.X, 2)) columns"))
  m.r .= m.y .- m.ybar
  for k in activecoordinates(x)
    v = x[k]
    iszero(v) || _column_axpy!(m.r, m.X, k, m.xbar[k], v)
  end
  fill!(m.a, T(NaN))
  m
end

# (a, s, ρ) with s = (X[:, k] - c)' r, ρ = r' r and a = ‖X[:, k] - c‖², which
# is computed only when the cached a is NaN
@inline function _column_dot_rsqr(X::AbstractMatrix{T}, r::AbstractVector{T}, k::Int, c::T, a::T) where {T}
  s = zero(T)
  ρ = zero(T)
  if isnan(a)
    a = zero(T)
    if iszero(c)
      @inbounds @simd for i in eachindex(r)
        a += X[i, k] * X[i, k]
        s += r[i] * X[i, k]
        ρ += r[i] * r[i]
      end
    else
      @inbounds @simd for i in eachindex(r)
        d = X[i, k] - c
        a += d * d
        s += r[i] * d
        ρ += r[i] * r[i]
      end
    end
  elseif iszero(c)
    @inbounds @simd for i in eachindex(r)
      s += r[i] * X[i, k]
      ρ += r[i] * r[i]
    end
  else
    @inbounds @simd for i in eachindex(r)
      s += r[i] * (X[i, k] - c)
      ρ += r[i] * r[i]
    end
  end
  a, s, ρ
end

# the exact minimizer of ‖r - X[:,k] t‖ + λ|t| over the coordinate, with the
# columns centered at their means when the model fits an intercept
function _step!(m::SqrtLSModel, g::NormL1, x::SparseIterate, k::Int)
  T = eltype(m)
  X, r = m.X, m.r
  c = m.xbar[k]

  # partial residuals with coordinate k removed: r = y - X*x + X[:,k]*x[k]
  old = x[k]
  iszero(old) || _column_axpy!(r, X, k, c, -old)

  xsqr, s, rsqr = _column_dot_rsqr(X, r, k, c, m.a[k])
  m.a[k] = xsqr
  λ = T(g.λ0 * _weight(g, k))

  # zero when the subgradient condition |X[:,k]'r| ≤ λ‖r‖ holds. Otherwise
  # λ² < ‖X[:,k]‖² and (X[:,k]'r)² ≤ ‖X[:,k]‖² ‖r‖² by Cauchy–Schwarz, but
  # rounding can break either when r is nearly parallel to X[:,k], so both are
  # enforced
  q = 1 - λ^2 / xsqr
  new = if abs(s) <= λ * sqrt(rsqr) || !(q > 0)
    zero(T)
  else
    (s - copysign(λ / sqrt(q) * sqrt(max(rsqr - s^2 / xsqr, zero(T))), s)) / xsqr
  end

  iszero(new) || _column_axpy!(r, X, k, c, new)
  new != old && (x[k] = new)
  new - old
end

_check_model(::SqrtLSModel, g) = g isa NormL1 ||
  throw(ArgumentError("coordinate descent on SqrtLeastSquaresLoss needs a NormL1 penalty, but g is a $(typeof(g))."))

function _sqnorm!(m::SqrtLSModel{T}, k::Int) where {T}
  a = m.a[k]
  if isnan(a)
    a = zero(T)
    X, c = m.X, m.xbar[k]
    @inbounds @simd for i in axes(X, 1)
      a += (X[i, k] - c) * (X[i, k] - c)
    end
    m.a[k] = a
  end
  a
end

_coordscale(m::SqrtLSModel, x, k::Int) = sqrt(_sqnorm!(m, k) / length(m.y))
function _responsescale(m::SqrtLSModel{T}, x) where {T}
  s = zero(T)
  @inbounds for i in eachindex(m.y)
    s += (m.y[i] - m.ybar)^2
  end
  sqrt(s / length(m.y))
end

_value(m::SqrtLSModel, x) = norm(m.r)

function gradient!(G::AbstractVector{T}, m::SqrtLSModel{T}, x) where {T}
  mul!(G, transpose(m.X), m.r)
  G .= (m.xbar .* sum(m.r) .- G) ./ norm(m.r)
end
