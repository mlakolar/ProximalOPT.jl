####################################
#
#  generalized linear losses  s ∑ᵢ wᵢ ℓ(ηᵢ, yᵢ),  η = Xβ + o
#
####################################

# The model of a GLMLoss whose scalar loss has a bounded curvature (Logistic,
# Huber): coordinate steps with the curvature bound s sup ℓ'' ∑ᵢ wᵢ Xᵢₖ²,
# which majorize the loss along the coordinate. It caches η, so a step costs
# O(n). ProxNewton converges much faster on these losses, and also handles a
# loss with an unbounded curvature (Poisson).
mutable struct GLMModel{T<:AbstractFloat, L, MX<:AbstractMatrix{T}, VY<:AbstractVector{T},
                        W<:Union{Nothing, AbstractVector{T}}, O<:Union{Nothing, AbstractVector{T}}} <: CDModel{T}
  loss::L
  X::MX
  y::VY
  w::W
  offset::O
  s::T
  bound::T              # sup ℓ''
  η::Vector{T}
  a::Vector{T}          # s sup ℓ'' ∑ᵢ wᵢ Xᵢₖ², NaN until computed
  d::Vector{T}          # wᵢ ℓ'(ηᵢ, yᵢ), for the gradient
end

function GLMModel(loss, y, X, w, offset, s)
  T = float(promote_type(eltype(y), eltype(X)))
  bound = T(ProximalBase.curvature(loss))
  isfinite(bound) || throw(ArgumentError("the curvature of $(typeof(loss)) is unbounded, so coordinate descent has no step size; use ProxNewton."))
  Xt, yt = convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y)
  wt = w === nothing ? nothing : convert(AbstractVector{T}, w)
  ot = offset === nothing ? nothing : convert(AbstractVector{T}, offset)
  _check_finite(yt, "y"); _check_finite(Xt, "X")
  n, p = size(Xt)
  GLMModel{T, typeof(loss), typeof(Xt), typeof(yt), typeof(wt), typeof(ot)}(loss, Xt, yt, wt, ot, T(s), bound,
                                                                             zeros(T, n), fill(T(NaN), p), zeros(T, n))
end

cdmodel(f::GLMLoss, x) = GLMModel(f.loss, f.y, f.X, f.weights, f.offset, 1)
cdmodel(f::Scaled{<:GLMLoss}, x) = GLMModel(f.f.loss, f.f.y, f.f.X, f.f.weights, f.f.offset, f.α)

@inline _obsweight(::Nothing, i::Int, ::Type{T}) where {T} = one(T)
@inline _obsweight(w::AbstractVector, i::Int, ::Type{T}) where {T} = @inbounds w[i]

function initialize!(m::GLMModel{T}, x::SparseIterate{T}) where {T}
  length(x) == size(m.X, 2) || throw(DimensionMismatch("x has length $(length(x)) and X has $(size(m.X, 2)) columns"))
  m.offset === nothing ? fill!(m.η, zero(T)) : copyto!(m.η, m.offset)
  for k in activecoordinates(x)
    v = x[k]
    iszero(v) || _column_axpy!(m.η, m.X, k, zero(T), -v)
  end
  fill!(m.a, T(NaN))
  m
end

function _curvaturebound!(m::GLMModel{T}, k::Int) where {T}
  a = m.a[k]
  if isnan(a)
    a = zero(T)
    X = m.X
    @inbounds for i in axes(X, 1)
      a += _obsweight(m.w, i, T) * X[i, k] * X[i, k]
    end
    a *= m.s * m.bound
    m.a[k] = a
  end
  a
end

@inline function coordinate_derivatives(m::GLMModel{T}, x::SparseIterate{T}, k::Int) where {T}
  X, η, y = m.X, m.η, m.y
  G = zero(T)
  @inbounds for i in eachindex(η)
    G += _obsweight(m.w, i, T) * T(ProximalBase.deriv(m.loss, η[i], y[i])) * X[i, k]
  end
  m.s * G, _curvaturebound!(m, k)
end

update_cache!(m::GLMModel{T}, x::SparseIterate{T}, k::Int, h::T) where {T} = (_column_axpy!(m.η, m.X, k, zero(T), -h); m)

# η = Xβ + o is affine in β, and the derivatives are computed from it
_cache(m::GLMModel) = m.η

coordinate_scale(m::GLMModel, x, k::Int) = sqrt(_curvaturebound!(m, k))

# as √(s ∑ᵢ wᵢ yᵢ²) for least squares: the scale of the derivatives at the
# offset, relative to the curvature
function response_scale(m::GLMModel{T}, x) where {T}
  v = zero(T)
  @inbounds for i in eachindex(m.y)
    oi = m.offset === nothing ? zero(T) : m.offset[i]
    v += _obsweight(m.w, i, T) * T(ProximalBase.deriv(m.loss, oi, m.y[i]))^2
  end
  sqrt(m.s * v / m.bound)
end

function loss_value(m::GLMModel{T}, x) where {T}
  v = zero(T)
  @inbounds for i in eachindex(m.η)
    v += _obsweight(m.w, i, T) * T(m.loss(m.η[i], m.y[i]))
  end
  m.s * v
end

function gradient!(G::AbstractVector{T}, m::GLMModel{T}, x) where {T}
  @inbounds for i in eachindex(m.η)
    m.d[i] = _obsweight(m.w, i, T) * T(ProximalBase.deriv(m.loss, m.η[i], m.y[i]))
  end
  mul!(G, transpose(m.X), m.d)
  G .*= m.s
end

####################################
#  proximal Newton
####################################

"""
    ProxNewton(; inner = CDSolver(; tol = 1e-8), maxiter = 100, tol = 1e-8, β = 0.5)

Proximal Newton for `f(x) + g(x)`, `f` a `GLMLoss` (or a scaled one) and `g`
a separable penalty, as glmnet fits generalized linear models: each step
minimizes the quadratic model of `f` at `x`, a weighted least-squares loss with
the weights `wᵢℓ''(ηᵢ)` and the working responses `ηᵢ - ℓ'(ηᵢ)/ℓ''(ηᵢ)`, plus
`g`, by coordinate descent (`inner`) warm started from `x`, and then backtracks
along the step on `f + g`, shrinking it by `β`, until the objective decreases.
It stops when the objective changes by less than `tol` relative to its value,
or after `maxiter` steps. Use with [`solve!`](@ref) on a `SparseIterate`;
returns [`CDResults`](@ref), whose `passes` sum the inner passes.

For the graphical lasso, `f` a `LogDetLoss(S, w)` (or a scaled one) and `g` a
`NormL1` (a matrix of weights leaves, for example, the diagonal free), it is
QUIC (Hsieh, Sustik, Dhillon and Ravikumar, *QUIC: quadratic approximation for
sparse inverse covariance estimation*, 2014), on a `SymmetricSparseIterate` or
a dense symmetric matrix `X`. Each step minimizes the second-order model
`w (tr(WΔWΔ)/2 + tr((S - W)Δ)) + g(X + Δ)`, `W = X⁻¹`, by coordinate descent
over the free set (the entries that are nonzero or whose gradient exceeds the
penalty), with up to `2 (1 + k ÷ 3)` passes at step `k`, and the Armijo line
search keeps `X` positive definite. The solution is block diagonal along the
connected components of the graph of the entries with `w |Sᵢⱼ|` above the
penalty, so each component is solved on its own. It stops when the relative
duality gap (see [`duality_gap`](@ref)) is at most `tol`, which the `gap` of
the results reports. `X` is the starting point when it is positive definite;
otherwise (for example zero) the start is the diagonal solution
`1/(Sᵢᵢ + λᵢᵢ/w)`.
"""
struct ProxNewton <: ProximalSolver
  inner::CDSolver
  maxiter::Int
  tol::Float64
  β::Float64
end

function ProxNewton(; inner::CDSolver = CDSolver(; tol = 1e-8), maxiter::Integer = 100, tol::Real = 1e-8,
                    β::Real = 0.5)
  maxiter >= 1 || throw(ArgumentError("maxiter must be positive, but it is $maxiter."))
  tol >= 0 || throw(ArgumentError("tol must be nonnegative, but it is $tol."))
  0 < β < 1 || throw(ArgumentError("β must be in (0, 1), but it is $β."))
  ProxNewton(inner, Int(maxiter), Float64(tol), Float64(β))
end

Base.summary(::ProxNewton) = "Proximal Newton"

_glm(f::GLMLoss) = (f, 1)
_glm(f::Scaled{<:GLMLoss}) = (f.f, f.α)
_glm(f) = throw(ArgumentError("ProxNewton needs a GLMLoss, but f is a $(typeof(f))."))

# s ∑ᵢ wᵢ ℓ(ηᵢ, yᵢ)
function _glmvalue(loss, y, w, s::T, η::Vector{T}) where {T}
  v = zero(T)
  @inbounds for i in eachindex(η)
    v += _obsweight(w, i, T) * T(loss(η[i], y[i]))
  end
  s * v
end

# x = x0 + t (x1 - x0)
function _interpolate!(x::SparseIterate{T}, x0::SparseIterate{T}, x1::SparseIterate{T}, t::T) where {T}
  copyto!(x, x0)
  for k in activecoordinates(x1)
    x[k] = x0[k] + t * (x1[k] - x0[k])
  end
  for k in activecoordinates(x0)
    isactive(x1, k) || (x[k] = (1 - t) * x0[k])
  end
  x
end

function solve!(x::SparseIterate{T}, f::AbstractFunction, g::AbstractFunction, method::ProxNewton) where {T}
  isseparable(g) || throw(ArgumentError("ProxNewton needs a separable penalty, but g is a $(typeof(g))."))
  glm, α = _glm(f)
  loss, y, X, w, o = glm.loss, glm.y, glm.X, glm.weights, glm.offset
  n, p = size(X)
  length(x) == p || throw(DimensionMismatch("x has length $(length(x)) and X has $p columns"))
  s = T(α)

  # η = Xx + o, and the objective at x
  η = zeros(T, n)
  o === nothing || copyto!(η, o)
  for k in activecoordinates(x)
    _column_axpy!(η, X, k, zero(T), -x[k])
  end
  F = _glmvalue(loss, y, w, s, η) + g(x)

  # the quadratic model at x: weights v, responses z (working responses minus
  # the offset), updated in place
  v, z = zeros(T, n), zeros(T, n)
  model = LSModel(z, X, v, nothing, s)
  η0, η1 = similar(η), similar(η)
  x0, x1 = copy(x), copy(x)
  passes, converged, inner_converged, iterations = 0, false, true, 0
  for it in 1:method.maxiter
    iterations = it
    @inbounds for i in 1:n
      d1 = T(ProximalBase.deriv(loss, η[i], y[i]))
      d2 = max(T(ProximalBase.deriv2(loss, η[i], y[i])), sqrt(eps(T)))
      v[i] = _obsweight(w, i, T) * d2
      z[i] = η[i] - d1 / d2 - (o === nothing ? zero(T) : o[i])
    end
    copyto!(x0, x)
    copyto!(η0, η)
    r = solve!(x, model, g, method.inner)
    passes += r.passes
    inner_converged &= r.converged
    copyto!(x1, x)
    # Xx + o at the full step, from the residuals z - Xx of the model
    @inbounds for i in 1:n
      η1[i] = z[i] - model.r[i] + (o === nothing ? zero(T) : o[i])
    end

    # backtracking along the step: x0 + t (x1 - x0), η0 + t (η1 - η0)
    t = one(T)
    copyto!(η, η1)
    Fnew = _glmvalue(loss, y, w, s, η) + g(x)
    while !(Fnew <= F + 10 * eps(T) * abs(F))
      t *= T(method.β)
      if t < 1e-10           # no decrease along the step: stay at x0
        copyto!(x, x0); copyto!(η, η0); Fnew = F
        break
      end
      _interpolate!(x, x0, x1, t)
      @. η = η0 + t * (η1 - η0)
      Fnew = _glmvalue(loss, y, w, s, η) + g(x)
    end
    dropzeros!(x)
    done = abs(F - Fnew) <= method.tol * max(abs(Fnew), one(T))
    F = Fnew
    if done
      converged = true
      break
    end
  end
  CDResults(x, F, passes, converged && inner_converged, T(NaN), T(method.tol), 0)
end
