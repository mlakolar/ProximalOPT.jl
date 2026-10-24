####################################
#
#  Duality gaps, gap-safe screening and optimality residuals
#
####################################

# For a loss of a linear map, f(x) = F(Ax) with F(η) = Σᵢ φᵢ(ηᵢ), and a convex
# penalty g with a closed-form conjugate, the Fenchel dual of minimizing
# P = f + g is D(θ) = -F*(θ) - g*(-Aᵀθ), and D(θ) ≤ P(x*) ≤ P(x) for every θ and
# x. The gap P(x) - D(θ) thus bounds the distance of P(x) to the optimum. The
# dual point is θ = t ∇F(Ax), scaled by t ∈ (0, 1] into the domain of g*; at the
# solution t = 1 and the gap is zero. The least-squares model has
# φᵢ(η) = s wᵢ (ỹᵢ - η)²/2 on the centered data ỹ = y - o - ȳ and A = X - 1x̄ᵀ,
# the GLM model φᵢ(η) = s wᵢ ℓ(η + oᵢ, yᵢ) and A = X; in both, ∇f(x) = Aᵀ∇F(Ax)
# is what gradient! returns.

####################################
#  conjugates
####################################

# the conjugate ℓ*(u; y) = sup_η uη - ℓ(η, y) of a scalar loss, Inf off its
# domain; the arguments come from u = t ℓ'(η, y), which is in the domain up to
# rounding
_conjugate(::ProximalBase.Squared, u::T, y) where {T} = u * u / 2 + u * T(y)
function _conjugate(::ProximalBase.Logistic, u::T, y) where {T}
  p = u + T(y)                 # the fitted probability
  _indomain(p, zero(T), one(T)) || return T(Inf)
  p = clamp(p, zero(T), one(T))
  _xlogx(p) + _xlogx(one(T) - p)
end
function _conjugate(::ProximalBase.Poisson, u::T, y) where {T}
  p = u + T(y)                 # the fitted mean
  _indomain(p, zero(T), T(Inf)) || return T(Inf)
  p = max(p, zero(T))
  _xlogx(p) - p
end
function _conjugate(ℓ::ProximalBase.Huber, u::T, y) where {T}
  δ = T(ℓ.δ)
  _indomain(u, -δ, δ) || return T(Inf)
  u * u / 2 + u * T(y)
end

_hasconjugate(ℓ) = false
_hasconjugate(::Union{ProximalBase.Squared, ProximalBase.Logistic, ProximalBase.Poisson, ProximalBase.Huber}) = true

# whether p lies in [a, b], up to rounding
@inline _indomain(p::T, a, b) where {T} = (tol = 8 * eps(T) * max(one(T), abs(p)); a - tol <= p <= b + tol)

# the penalties with a closed-form conjugate: the dual point is feasible after a
# scaling by t, and g*(-t∇f) has a closed form
_hasdual(g) = false
_hasdual(::Union{NormL1, NormL1Nonneg, ElasticNet, GroupNormL2}) = true

# The largest t ∈ (0, 1] for which -t z is in the domain of g*, for z = ∇f(x):
# |t zᵢ| ≤ λ wᵢ for the ℓ1 norm, -t zᵢ ≤ λ wᵢ on the nonnegative orthant,
# ‖t z_G‖ ≤ λ w_G for the groups. nothing when a coordinate without penalty has
# a nonzero gradient, which no scaling fixes.
function _dualscale(g::NormL1, z::AbstractVector{T}) where {T}
  t = one(T)
  @inbounds for i in eachindex(z)
    a, c = abs(z[i]), T(g.λ0) * T(_entryweight(g.w, i, T))
    a > c || continue
    iszero(c) && return nothing
    t = min(t, c / a)
  end
  t
end

function _dualscale(g::NormL1Nonneg, z::AbstractVector{T}) where {T}
  t = one(T)
  @inbounds for i in eachindex(z)
    a, c = -z[i], T(g.λ0) * T(_entryweight(g.w, i, T))
    a > c || continue
    iszero(c) && return nothing
    t = min(t, c / a)
  end
  t
end

# with a ridge term, g* is finite everywhere and needs no scaling
_dualscale(g::ElasticNet, z::AbstractVector{T}) where {T} =
  iszero(g.λ2) ? _dualscale(NormL1(g.λ1, g.w), z) : one(T)

function _dualscale(g::GroupNormL2, z::AbstractVector{T}) where {T}
  for i in eachindex(z)        # an index in no group is not penalized
    (i > length(g.group_of) || iszero(g.group_of[i])) && !iszero(z[i]) && return nothing
  end
  t = one(T)
  for (k, G) in enumerate(g.groups)
    a = zero(T)
    @inbounds for i in G
      a += z[i] * z[i]
    end
    a, c = sqrt(a), T(g.λ0 * g.w[k])
    a > c || continue
    iszero(c) && return nothing
    t = min(t, c / a)
  end
  t
end

# g*(-t z), which is zero for the norms once t has scaled the point into the
# domain, and Σᵢ (|t zᵢ| - λ1 wᵢ)₊² / (2λ2) for the elastic net
_conjugate_penalty(g, z::AbstractVector{T}, t::T) where {T} = zero(T)
function _conjugate_penalty(g::ElasticNet, z::AbstractVector{T}, t::T) where {T}
  iszero(g.λ2) && return zero(T)
  v = zero(T)
  @inbounds for i in eachindex(z)
    d = abs(t * z[i]) - T(g.λ1) * T(_entryweight(g.w, i, T))
    d > 0 && (v += d * d)
  end
  v / (2 * T(g.λ2))
end

####################################
#  the gaps of the models
####################################

# whether the model m and the penalty g have a duality gap
_hasgap(m::CDModel, g) = false
_hasgap(m::LSModel, g) = _hasdual(g)
_hasgap(m::GLMModel, g) = _hasdual(g) && _hasconjugate(m.loss)

# (P(x), D(tθ), t) for a model current for x, writing ∇f(x) into z; nothing
# without a gap. For least squares θᵢ = -s wᵢ rᵢ with the residuals r = ỹ - Ax
# (m.r .+ m.δ, see LSModel), so F*(tθ) = Σᵢ -t s wᵢ rᵢ ỹᵢ + t² s wᵢ rᵢ²/2.
_gap!(z, m::CDModel, g, x) = nothing

function _gap!(z::AbstractVector{T}, m::LSModel{T}, g, x) where {T}
  _hasdual(g) || return nothing
  gradient!(z, m, x)
  t = _dualscale(g, z)
  t === nothing && return nothing
  r, δ, y, o, w, ȳ = m.r, m.δ, m.y, m.offset, m.w, m.ybar
  v = zero(T)
  @inbounds for i in eachindex(r)
    wi = w === nothing ? one(T) : w[i]
    ỹ = (o === nothing ? y[i] : y[i] - o[i]) - ȳ
    ri = r[i] + δ
    v += wi * ri * (t * ri / 2 - ỹ)
  end
  P = loss_value(m, x) + g(x)
  P, -t * m.s * v - _conjugate_penalty(g, z, t), t
end

function _gap!(z::AbstractVector{T}, m::GLMModel{T}, g, x) where {T}
  _hasgap(m, g) || return nothing
  gradient!(z, m, x)
  t = _dualscale(g, z)
  t === nothing && return nothing
  P = loss_value(m, x) + g(x)
  P, -_glm_conjugate(m.loss, m.y, m.w, m.offset, m.s, m.η, t) - _conjugate_penalty(g, z, t), t
end

# F*(tθ) for θᵢ = s wᵢ ℓ'(ηᵢ, yᵢ), η = Ax + o: Σᵢ s wᵢ (ℓ*(uᵢ; yᵢ) - uᵢ oᵢ) with
# uᵢ = t ℓ'(ηᵢ, yᵢ); an observation of weight zero adds nothing
function _glm_conjugate(loss, y, w, o, s::T, η::AbstractVector{T}, t::T) where {T}
  v = zero(T)
  @inbounds for i in eachindex(η)
    wi = _obsweight(w, i, T)
    iszero(wi) && continue
    u = t * T(ProximalBase.deriv(loss, η[i], y[i]))
    c = _conjugate(loss, u, y[i])
    isfinite(c) || return T(Inf)
    v += wi * (c - u * (o === nothing ? zero(T) : T(o[i])))
  end
  s * v
end

####################################
#  the gap in coordinate descent
####################################

# the gap state of a run, created on first use and kept in st for the levels of
# a path; nothing when neither gaptol nor gapsafe asks for it or the model and
# penalty have no gap. Screening starts again in each call of _cd!.
function _gapstart!(st::_CDState{T}, m, g, x, s::CDSolver) where {T}
  st.relgap = T(NaN)
  (s.gaptol > 0 || s.gapsafe) || return nothing
  if !_hasgap(m, g)
    s.gaptol > 0 && @warn "there is no duality gap for $(nameof(typeof(m))) with $(nameof(typeof(g))); gaptol is ignored." maxlog = 1
    return nothing
  end
  gs = st.gap
  n = ncoordinates(x)
  if gs === nothing || length(gs.z) != length(x) || length(gs.keep) != n
    gs = _GapState{T}(Vector{T}(undef, length(x)), trues(n), sizehint!(Int[], n), nothing)
    st.gap = gs
  end
  fill!(gs.keep, true)
  gs.base = st.candidates
  gs
end

# The decision after a pass: (stop, converged, tol, ok), where converged says
# whether the changes were below tol (so the next pass is a full one) and ok
# whether the stopping rule was met. Without a gap, the run stops after a full
# pass with changes below tol. With gaptol, the gap is computed where that rule
# would stop, a full pass with changes below tol: the run stops if the gap is
# small enough, and tol is tightened tenfold otherwise. A gap costs a full
# gradient, which can cost much more than a pass over the candidates of the
# strong rule, so it is not computed after the other full passes, except for
# the screening of gapsafe, which pays only if it comes early.
@inline function _afterpass!(::Nothing, x, m, g, st, s::CDSolver, full::Bool, maxh, tol, tolmin)
  converged = maxh < tol
  full && converged, converged, tol, true
end

function _afterpass!(gs::_GapState{T}, x, m, g, st::_CDState{T}, s::CDSolver, full::Bool, maxh::T, tol::T,
                     tolmin::T) where {T}
  converged = maxh < tol
  (full && (converged || s.gapsafe)) || return false, converged, tol, true
  available, certified, moved = _gapcheck!(gs, x, m, g, st, s)
  certified && return true, converged, tol, true
  moved && return false, false, tol, true        # screening changed x: pass again
  converged || return false, converged, tol, true
  (s.gaptol > 0 && available) || return true, converged, tol, true
  tol <= tolmin && return true, converged, tol, false    # tol cannot be tightened further
  tol = max(tol / 10, tolmin)
  false, maxh < tol, tol, true
end

# the relative duality gap at x, recorded in st, and the gap-safe screening;
# returns (available, certified, moved): whether the gap could be computed,
# whether it is at most gaptol, and whether screening changed x
function _gapcheck!(gs::_GapState{T}, x, m, g, st::_CDState{T}, s::CDSolver) where {T}
  res = _gap!(gs.z, m, g, x)
  if res === nothing
    st.relgap = T(NaN)
    return false, false, false
  end
  P, D, t = res
  gap = max(P - D, zero(T))
  st.relgap = gap / max(abs(P), floatmin(T))
  certified = s.gaptol > 0 && st.relgap <= s.gaptol
  moved = s.gapsafe && !certified && _gapscreen!(gs, m, g, x, st, t, gap)
  true, certified, moved
end

# Gap-safe screening (Fercoq, Gramfort and Salmon, 2015; Ndiaye, Fercoq,
# Gramfort and Salmon, 2017). Each φᵢ has curvature at most cᵢ, so the dual is
# strongly concave in the norm ‖θ‖² = Σᵢ θᵢ²/cᵢ and its maximizer θ* is within
# √(2 gap) of θ in that norm; then |Aₖᵀ(θ - θ*)| ≤ √(2 gap aₖ), with
# aₖ = Σᵢ cᵢ Aᵢₖ² the curvature of f along coordinate k that coordinate descent
# already uses, and the largest eigenvalue of A_Gᵀ C A_G, the curvature over the
# group, for a group. A coordinate with |Aₖᵀθ| + √(2 gap aₖ) < λ wₖ (with the
# gradient in place of |Aₖᵀθ| for the nonnegative ℓ1 norm, and its norm over a
# group) is zero at the solution: it is set to zero and left out of the full
# passes for the rest of the run. Returns whether x changed.
function _gapscreen!(gs::_GapState{T}, m, g, x, st::_CDState{T}, t::T, gap::T) where {T}
  z, keep = gs.z, gs.keep
  moved, screened = false, false
  for k in eachindex(keep)
    keep[k] || continue
    c = T(_level(g)) * T(_coordweight(g, x, k))
    c > 0 || continue
    a = t * _screenstat(g, z, x, k)
    a < c || continue
    a + sqrt(2 * gap * _screencurvature(m, x, k)) < c || continue
    keep[k] = false
    screened = true
    moved |= _zerocoordinate!(m, x, k)
  end
  moved && dropzeros!(x)
  screened && _setcandidates!(gs, st)
  moved
end

_screenstat(g, z::AbstractVector, x, k::Int) = _gradnorm(z, x, k)
_screenstat(g::NormL1Nonneg, z::AbstractVector, x::SparseIterate, k::Int) = -z[k]

_screencurvature(m::LSModel, x::SparseIterate, k::Int) = m.s * _sqnorm!(m, k)
_screencurvature(m::LSModel, x::AtomIterate, k::Int) = _groupcurvature!(m, x, k)
_screencurvature(m::GLMModel, x::SparseIterate, k::Int) = _curvaturebound!(m, k)

# sets coordinate k of x to zero and updates the cache; returns whether it moved
function _zerocoordinate!(m, x::SparseIterate{T}, k::Int) where {T}
  v = x[k]
  iszero(v) && return false
  x[k] = zero(T)
  update_cache!(m, x, k, -v)
  true
end

function _zerocoordinate!(m::LSModel{T}, x::AtomIterate{T}, k::Int) where {T}
  isactive(x, k) || return false
  moved = false
  for j in x.groups[k]
    v = x[j]
    iszero(v) && continue
    x[j] = zero(T)
    _residual_axpy!(m, j, -v)
    moved = true
  end
  moved
end

# the candidates of the full passes: those on entry to _cd! (all coordinates
# when there were none) that screening kept
function _setcandidates!(gs::_GapState, st::_CDState)
  cand, keep = gs.cand, gs.keep
  empty!(cand)
  if gs.base === nothing
    for k in eachindex(keep)
      keep[k] && push!(cand, k)
    end
  else
    for k in gs.base
      keep[k] && push!(cand, k)
    end
  end
  st.candidates = cand
  gs
end

####################################
#  the gap of the functions of ProximalBase
####################################

# Evaluators of the gap for the gradient solvers, ADMM and duality_gap: the
# least-squares model of a least-squares loss, which holds the residuals, the
# predictor η = Xx + o and the gradient of a GLM, and the workspaces of the
# graphical lasso. _gapevaluator(f, g, x) is nothing without a gap.
struct _LSGap{M<:LSModel, T}
  m::M
  z::Vector{T}
end

struct _GLMGap{L, MX, VY, W, O, T}
  loss::L
  X::MX
  y::VY
  w::W
  o::O
  s::T
  η::Vector{T}
  d::Vector{T}
  z::Vector{T}
end

struct _LogDetGap{T}
  C::Matrix{T}       # the Cholesky factor of x, then the upper triangle of x⁻¹
  M::Matrix{T}       # S + U and its Cholesky factor
end

const _LSLoss = Union{LeastSquaresLoss{<:Any, <:AbstractVector}, Scaled{<:LeastSquaresLoss{<:Any, <:AbstractVector}},
                      GLMLoss{ProximalBase.Squared}, Scaled{<:GLMLoss{ProximalBase.Squared}}}

_gapevaluator(f, g, x) = nothing

function _gapevaluator(f::_LSLoss, g, x::AbstractVector)
  _hasdual(g) || return nothing
  m = cdmodel(f, x)
  _LSGap(m, zeros(eltype(m), length(x)))
end

# the GLMLoss of f and its scale
_glm(f::GLMLoss) = (f, 1)
_glm(f::Scaled{<:GLMLoss}) = (f.f, f.α)

function _gapevaluator(f::Union{GLMLoss, Scaled{<:GLMLoss}}, g, x::AbstractVector)
  glm, α = _glm(f)
  (_hasdual(g) && _hasconjugate(glm.loss)) || return nothing
  n, p = size(glm.X)
  T = float(promote_type(eltype(glm.X), eltype(glm.y), eltype(x)))
  _GLMGap(glm.loss, glm.X, glm.y, glm.weights, glm.offset, T(α), zeros(T, n), zeros(T, n), zeros(T, p))
end

function _gapevaluator(f::LogDetLoss{T}, g::NormL1, x::AbstractMatrix) where {T}
  p = size(f.S, 1)
  _LogDetGap(zeros(T, p, p), zeros(T, p, p))
end

# v .+= α X x, touching only the stored entries of a sparse iterate
function _addproduct!(v::AbstractVector{T}, X::AbstractMatrix, x::SparseIterate, α::T) where {T}
  for k in activecoordinates(x)
    xk = x[k]
    iszero(xk) || _column_axpy!(v, X, k, zero(T), -α * xk)
  end
  v
end
function _addproduct!(v::AbstractVector{T}, X::AbstractMatrix, x::AtomIterate, α::T) where {T}
  for k in _activeentries(x)
    xk = x[k]
    iszero(xk) || _column_axpy!(v, X, k, zero(T), -α * xk)
  end
  v
end
_addproduct!(v::AbstractVector{T}, X::AbstractMatrix, x::AbstractVector, α::T) where {T} = mul!(v, X, x, α, one(T))

# (P(x), D) at x; D is -Inf when the dual point is not feasible
function _evalgap!(ev::_LSGap, g, x)
  m = ev.m
  r = m.r
  m.offset === nothing ? (r .= m.y) : (r .= m.y .- m.offset)     # no intercept: ȳ = 0, x̄ = 0
  _addproduct!(r, m.X, x, -one(eltype(r)))
  res = _gap!(ev.z, m, g, x)
  res === nothing ? nothing : (res[1], res[2])
end

function _evalgap!(ev::_GLMGap{L, MX, VY, W, O, T}, g, x) where {L, MX, VY, W, O, T}
  η, d = ev.η, ev.d
  ev.o === nothing ? fill!(η, zero(T)) : (η .= ev.o)
  _addproduct!(η, ev.X, x, one(T))
  @inbounds for i in eachindex(η)
    d[i] = ev.s * _obsweight(ev.w, i, T) * T(ProximalBase.deriv(ev.loss, η[i], ev.y[i]))
  end
  mul!(ev.z, transpose(ev.X), d)
  t = _dualscale(g, ev.z)
  t === nothing && return nothing
  P = _glmvalue(ev.loss, ev.y, ev.w, ev.s, η) + g(x)
  P, -_glm_conjugate(ev.loss, ev.y, ev.w, ev.o, ev.s, η, t) - _conjugate_penalty(g, ev.z, t)
end

# The graphical lasso P(X) = w (tr(SX) - log det X) + Σᵢⱼ λᵢⱼ |Xᵢⱼ| has the dual
# max w (log det(S + U) + p) over symmetric U with |Uᵢⱼ| ≤ (λᵢⱼ + λⱼᵢ)/(2w); the
# dual point is U = W - S with W = X⁻¹, clipped to these bounds, so that
# S + U = W at the solution.
function _evalgap!(ev::_LogDetGap{T}, f::LogDetLoss{T}, g::NormL1, X::AbstractMatrix) where {T}
  P = f(X) + g(X)
  isfinite(P) || return P, T(-Inf)
  C, M = ev.C, ev.M
  copyto!(C, X)
  _logdet_chol!(C) === nothing && return P, T(-Inf)
  _inverse_from_chol!(M, C)
  P, _glasso_dual!(C, f.S, M, g, f.w)
end

# the bound on the dual variable of entry (r, c) over symmetric matrices, the
# mean of the penalties of (r, c) and (c, r), divided by w
@inline _glasso_bound(g::NormL1, p::Int, r::Int, c::Int, w::T) where {T} =
  T(g.λ0) * (T(_entryweight(g.w, (c - 1) * p + r, T)) + T(_entryweight(g.w, (r - 1) * p + c, T))) / (2w)

# log det of the positive definite matrix whose upper triangle is factorized in
# place in C, or nothing
function _logdet_chol!(C::Matrix{T}) where {T}
  _, info = LAPACK.potrf!('U', C)
  info == 0 || return nothing
  ld = zero(T)
  @inbounds for i in axes(C, 1)
    ld += log(C[i, i])
  end
  2ld
end

# W = X⁻¹ from the Cholesky factor of X in C (upper triangle), as a full
# symmetric matrix
function _inverse_from_chol!(W::Matrix{T}, C::Matrix{T}) where {T}
  LAPACK.potri!('U', C)
  @inbounds for c in axes(C, 2), r in 1:c
    W[r, c] = C[r, c]
    W[c, r] = C[r, c]
  end
  W
end

# the dual value w (log det(S + U) + p) at U = W - S clipped to the bounds, with
# M as the workspace; -Inf when S + U is not positive definite
function _glasso_dual!(M::Matrix{T}, S::AbstractMatrix{T}, W::Matrix{T}, g::NormL1, w::T) where {T}
  p = size(S, 1)
  @inbounds for c in 1:p, r in 1:c
    M[r, c] = S[r, c] + clamp(W[r, c] - S[r, c], -_glasso_bound(g, p, r, c, w), _glasso_bound(g, p, r, c, w))
  end
  ld = _logdet_chol!(M)
  ld === nothing ? T(-Inf) : w * (ld + p)
end

_evalgap!(ev, f, g, x) = _evalgap!(ev, g, x)

# the relative gap (P - D)/|P| at x
function _relgap!(ev, f, g, x)
  res = _evalgap!(ev, f, g, x)
  res === nothing && return NaN
  P, D = res
  Float64(max(P - D, zero(P)) / max(abs(P), floatmin(typeof(P))))
end

"""
    duality_gap(f, g, x) -> (gap, primal, dual)

The duality gap `primal - dual` of minimizing `f + g` at `x`, where `primal`
is `f(x) + g(x)` and `dual` the value of the Fenchel dual at a dual point built
from the gradient at `x`, so that `0 ≤ primal - min(f + g) ≤ gap`: a
certificate of the accuracy of `x`, which is zero at the solution. The dual
point is `θ = t ∇F(Ax)` for `f(x) = F(Ax)`, scaled by `t ∈ (0, 1]` into the
domain of the conjugate of `g` (Fercoq, Gramfort and Salmon, 2015).

Available for:
- `f` a `LeastSquaresLoss` with a vector `y`, or a `GLMLoss` with the squared,
  logistic, Poisson or Huber loss, possibly scaled, and `g` one of `NormL1`,
  `NormL1Nonneg` and `ElasticNet` with positive weights, or a `GroupNormL2`
  with positive weights whose groups cover `x`. Here `x` is a vector, a
  `SparseIterate` or, for a `GroupNormL2`, an `AtomIterate`. A coordinate
  without penalty has no dual bound, so with one the gap exists only where its
  gradient is zero.
- `f` a [`CDModel`](@ref) of such a loss, as from [`cdmodel`](@ref), for
  instance with an intercept; `duality_gap` initializes it at `x`.
- the graphical lasso: `f` a `LogDetLoss`, `g` a `NormL1` (with a matrix of
  weights, for example a zero diagonal) and `x` a symmetric matrix. The dual
  point is `W - S` clipped to the penalty, with `W = x⁻¹`, at the cost of two
  Cholesky factorizations.

An `ArgumentError` is thrown for other functions; see [`kkt_residual`](@ref)
for an optimality measure that needs no conjugate.
"""
function duality_gap(f::AbstractFunction, g::AbstractFunction, x)
  ev = _gapevaluator(f, g, x)
  ev === nothing && throw(ArgumentError("no duality gap is known for $(nameof(typeof(f))) with $(nameof(typeof(g))) on a $(nameof(typeof(x))); use kkt_residual."))
  res = _evalgap!(ev, f, g, x)
  res === nothing && throw(ArgumentError("the dual point is not feasible: a coordinate without penalty has a nonzero gradient."))
  P, D = res
  (gap = P - D, primal = P, dual = D)
end

function duality_gap(m::CDModel{T}, g::AbstractFunction, x) where {T}
  _hasgap(m, g) || throw(ArgumentError("no duality gap is known for $(nameof(typeof(m))) with $(nameof(typeof(g))); use kkt_residual."))
  initialize!(m, x)
  res = _gap!(Vector{T}(undef, length(x)), m, g, x)
  res === nothing && throw(ArgumentError("the dual point is not feasible: a coordinate without penalty has a nonzero gradient."))
  P, D, _ = res
  (gap = P - D, primal = P, dual = D)
end

"""
    kkt_residual(f, g, x; relative = true)

The largest scaled change `√aₖ ⋅ |Δxₖ|` that one coordinate step of
[`CDSolver`](@ref) would make at `x`, over all coordinates, where `aₖ` is the
curvature of `f` along coordinate `k` and the step is the proximal step from
[`coordinate_derivatives`](@ref). It is zero exactly at a minimizer of `f + g`
when both are convex, and it is the quantity that the stopping rule of
`CDSolver` compares with `tol`: with `relative`, it is divided by the scale of
the data ([`response_scale`](@ref)), as there. `x` is not changed.

It applies to the losses that [`cdmodel`](@ref) knows and that define
`coordinate_derivatives` (`LeastSquaresLoss`, `GLMLoss`, `QuadraticFunction`,
[`KronQuadratic`](@ref)), or to a [`CDModel`](@ref), on a `SparseIterate` or a
`SymmetricSparseIterate`, with any separable penalty. For the losses without a
duality gap (see [`duality_gap`](@ref)), such as the quadratics, it is the
optimality measure to use.
"""
function kkt_residual(f::Union{AbstractFunction, CDModel}, g::AbstractFunction,
                      x::Union{SparseIterate, SymmetricSparseIterate}; relative::Bool = true)
  m = f isa CDModel ? f : cdmodel(f, x)
  hasmethod(coordinate_derivatives, Tuple{typeof(m), typeof(x), Int}) ||
    throw(ArgumentError("kkt_residual needs coordinate_derivatives for $(nameof(typeof(m))) on a $(nameof(typeof(x)))."))
  _check_iterate(g, x)
  check_penalty(m, g)
  initialize!(m, x)
  T = eltype(m)
  res = zero(T)
  for k in 1:ncoordinates(x)
    res = max(res, _fixedpoint_residual(m, g, x, k))
  end
  sy = relative ? response_scale(m, x) : zero(T)
  sy > 0 ? res / sy : res
end

# √a |prox step| at coordinate k, without moving it
function _fixedpoint_residual(m::CDModel{T}, g, x::SparseIterate{T}, k::Int) where {T}
  G, a = coordinate_derivatives(m, x, k)
  iszero(a) && return zero(T)
  old = x[k]
  sqrt(a) * abs(prox_entry(g, old - G / a, k, inv(a)) - old)
end

function _fixedpoint_residual(m::CDModel{T}, g, x::SymmetricSparseIterate{T}, k::Int) where {T}
  G, a = coordinate_derivatives(m, x, k)
  iszero(a) && return zero(T)
  r, c = tril_cartesian(x.p, k)
  old = coordinates(x)[k]
  sqrt(a) * abs(prox_entry(g, old - G / a, _entry(x.p, r, c), _multiplicity(T, r, c) / a) - old)
end
