# Functions the benchmark needs that ProximalBase 0.5.2 (the registered version)
# doesn't have. They are local to the benchmark, written for these problems only.

using LinearAlgebra, ProximalBase, ProximalCore

# The quantile (check) loss of the residuals y - z, scaled by c:
# h(z) = c Σᵢ ρ_τ(yᵢ - zᵢ), with ρ_τ(u) = max(τu, (τ - 1)u). ProximalBase's
# CheckLoss is in an open PR, so this is the ProximalCore interface only (the
# primal–dual algorithms use the prox of h, and of h* by Moreau's identity).
struct CheckResidual{T}
  y::Vector{T}
  τ::T
  c::T
end
_check(u, τ) = u >= 0 ? τ * u : (τ - 1) * u
(h::CheckResidual)(z) = h.c * sum(i -> _check(h.y[i] - z[i], h.τ), eachindex(h.y))

# prox_{γh}(v) = y - prox_{γcρ_τ}(y - v), with the prox of tρ_τ at w equal to
# w - tτ for w > tτ, w + t(1 - τ) for w < -t(1 - τ), and 0 otherwise
function ProximalCore.prox!(z, h::CheckResidual, v, γ)
  t, τ = γ * h.c, h.τ
  val = zero(eltype(z))
  @inbounds for i in eachindex(z)
    w = h.y[i] - v[i]
    u = w > t * τ ? w - t * τ : (w < -t * (1 - τ) ? w + t * (1 - τ) : zero(w))
    z[i] = h.y[i] - u
    val += _check(u, τ)
  end
  return h.c * val
end
ProximalCore.is_convex(::Type{<:CheckResidual}) = true

# The D-trace loss of a symmetric Δ, f(Δ) = tr(AΔBΔ)/2 + tr(CΔ), with its
# gradient over symmetric matrices, (AΔB + BΔA)/2 + C, for symmetric A, B, C.
# ProximalBase's KronQuadratic has this gradient only on a SymmetricSparseIterate;
# this is the dense version, so that both packages' solvers run on a Matrix.
struct SymKron{T} <: ProximalBase.AbstractFunction
  A::Matrix{T}
  B::Matrix{T}
  C::Matrix{T}
  W::Matrix{T}          # AΔ
  V::Matrix{T}          # AΔB
end
SymKron(A, B, C) = SymKron(Matrix(A), Matrix(B), Matrix(C), zeros(size(A)), zeros(size(A)))
ProximalBase.issmooth(::Type{<:SymKron}) = true
ProximalBase.isconvex(::Type{<:SymKron}) = true

function ProximalBase.value_and_gradient!(G::AbstractMatrix, f::SymKron, Δ::AbstractMatrix)
  mul!(f.W, f.A, Δ)
  mul!(G, f.W, f.B)                  # AΔB
  v = dot(Δ, G) / 2 + dot(f.C, Δ)
  p = size(G, 1)
  @inbounds for j in 1:p, i in 1:j   # G = (AΔB + (AΔB)')/2 + C, since BΔA = (AΔB)'
    s = (G[i, j] + G[j, i]) / 2
    G[i, j] = s + f.C[i, j]
    G[j, i] = s + f.C[j, i]
  end
  v
end
ProximalBase.gradient!(G::AbstractMatrix, f::SymKron, Δ::AbstractMatrix) = (ProximalBase.value_and_gradient!(G, f, Δ); G)
# tr(AΔBΔ) = ⟨Δ, AΔB⟩ for symmetric Δ
function (f::SymKron)(Δ::AbstractMatrix)
  mul!(f.W, f.A, Δ)
  mul!(f.V, f.W, f.B)
  dot(Δ, f.V) / 2 + dot(f.C, Δ)
end

# At most s nonzero entries in the strict lower triangle of a symmetric matrix,
# the diagonal free: ProximalBase's IndBallL0(s; diagonal = false) applied to
# the symmetric matrix, whose method needs a Symmetric argument.
struct SymL0 <: ProximalBase.AbstractFunction
  ball::IndBallL0
end
SymL0(s::Integer) = SymL0(IndBallL0(s; diagonal = false))
ProximalBase.isproximable(::Type{SymL0}) = true
(g::SymL0)(X::AbstractMatrix) = g.ball(Symmetric(X, :L))
function ProximalBase.value_and_prox!(Y::AbstractMatrix, g::SymL0, X::AbstractMatrix, γ::Real)
  ProximalBase.prox!(Y, g.ball, Symmetric(X, :L), γ)
  zero(eltype(Y))
end
ProximalBase.prox!(Y::AbstractMatrix, g::SymL0, X::AbstractMatrix, γ::Real) = (ProximalBase.value_and_prox!(Y, g, X, γ); Y)
