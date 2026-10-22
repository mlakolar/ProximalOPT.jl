####################################
#
#  quadratic  s (x'Ax/2 + b'x + c)
#
####################################

# The model of QuadraticFunction, possibly scaled by s. It caches Ax, so a
# coordinate step costs O(p). A is kept as a dense matrix: indexing a Symmetric
# branches on every access, which would dominate the updates.
mutable struct QuadModel{T<:AbstractFloat, B<:Union{Nothing, AbstractVector{T}}} <: CDModel{T}
  A::Matrix{T}
  b::B
  c::T
  s::T
  u::Vector{T}          # A x
  L::Vector{T}          # curvatures of the groups of an AtomIterate, NaN until computed
  tmp::Vector{T}
end

function QuadModel(A::AbstractMatrix, b, c, s)
  T = float(eltype(A))
  n = LinearAlgebra.checksquare(A)
  b === nothing || length(b) == n || throw(DimensionMismatch("b has length $(length(b)) and A has $n columns"))
  Ad = Matrix{T}(A)
  _check_finite(Ad, "A")
  bt = b === nothing ? nothing : convert(AbstractVector{T}, b)
  bt === nothing || _check_finite(bt, "b")
  QuadModel{T, typeof(bt)}(Ad, bt, T(c), T(s), zeros(T, n), T[], T[])
end

cdmodel(f::QuadraticFunction, x) = QuadModel(f.A, f.b, f.c, 1)
cdmodel(f::Scaled{<:QuadraticFunction}, x) = QuadModel(f.f.A, f.f.b, f.f.c, f.α)

@inline _linear(b::Nothing, k::Int, ::Type{T}) where {T} = zero(T)
@inline _linear(b::AbstractVector, k::Int, ::Type{T}) where {T} = @inbounds b[k]

function initialize!(m::QuadModel{T}, x::Union{SparseIterate{T}, AtomIterate{T}}) where {T}
  length(x) == size(m.A, 2) || throw(DimensionMismatch("x has length $(length(x)) and A has $(size(m.A, 2)) columns"))
  u, A = m.u, m.A
  fill!(u, zero(T))
  for k in _activeentries(x)
    v = x[k]
    iszero(v) && continue
    @inbounds @simd for i in eachindex(u)
      u[i] += A[i, k] * v
    end
  end
  x isa AtomIterate && fill!(resize!(m.L, ncoordinates(x)), T(NaN))
  m
end

# a coordinate with A[k, k] = 0, which f does not depend on when A is positive
# semidefinite, goes to the minimizer of the penalty nearest zero
@inline function _step!(m::QuadModel{T}, g, x::SparseIterate{T}, k::Int) where {T}
  A, u = m.A, m.u
  ainv = inv(m.s * @inbounds(A[k, k]))
  G = m.s * (@inbounds(u[k]) + _linear(m.b, k, T))
  old = x[k]
  new = isinf(ainv) ? cdprox(g, zero(T), k, floatmax(T)) : cdprox(g, old - G * ainv, k, ainv)
  h = new - old
  iszero(h) && return h      # x and Ax do not change
  x[k] = new
  @inbounds @simd for i in eachindex(u)
    u[i] += A[i, k] * h
  end
  h
end

@inline _coordinate(m::QuadModel{T}, x::SparseIterate{T}, k::Int) where {T} =
  m.s * (@inbounds(m.u[k]) + _linear(m.b, k, T)), m.s * @inbounds(m.A[k, k])

function _update!(m::QuadModel{T}, x::SparseIterate{T}, k::Int, h::T) where {T}
  A, u = m.A, m.u
  @inbounds @simd for i in eachindex(u)
    u[i] += A[i, k] * h
  end
  m
end

# with A = X'X/n and b = -X'y/n, √A[k,k] = ‖X[:, k]‖ / √n, and
# maxₖ |bₖ| / √A[k,k] is a lower bound on ‖y‖ / √n
_coordscale(m::QuadModel, x, k::Int) = sqrt(m.s * m.A[k, k])

function _responsescale(m::QuadModel{T}, x) where {T}
  m.b === nothing && return zero(T)
  s = zero(T)
  for k in eachindex(m.b)
    a = m.A[k, k]
    a > 0 && (s = max(s, abs(m.b[k]) / sqrt(a)))
  end
  sqrt(m.s) * s
end

function _value(m::QuadModel{T}, x) where {T}
  v = zero(T)
  for k in _activeentries(x)
    xk = x[k]
    v += xk * (m.u[k] / 2 + _linear(m.b, k, T))
  end
  m.s * (v + m.c)
end

gradient!(G::AbstractVector{T}, m::QuadModel{T}, x) where {T} =
  m.b === nothing ? (G .= m.s .* m.u) : (G .= m.s .* (m.u .+ m.b))

function _groupcurvature!(m::QuadModel{T}, x::AtomIterate{T}, k::Int) where {T}
  L = m.L[k]
  if isnan(L)
    G = x.groups[k]
    L = length(G) == 1 ? m.A[first(G), first(G)] : eigmax(Symmetric(m.A[G, G]))
    L = max(m.s * L, zero(T))
    m.L[k] = L
  end
  L
end

# group k of x ← the proximal operator of λ0 wₖ ‖⋅‖₂ with step 1/L at
# x[G] - ∇_G f(x) / L, as for least squares
function _groupstep!(m::QuadModel{T}, g::GroupNormL2, x::AtomIterate{T}, k::Int, scaled::Bool) where {T}
  A, u = m.A, m.u
  G = x.groups[k]
  c = T(g.λ0 * g.w[k])
  length(m.tmp) < length(G) && resize!(m.tmp, length(G))
  v = m.tmp

  # v = ∇_G f(x) = s (Ax + b)[G]; a group at zero stays there if ‖v‖ ≤ c
  t = zero(T)
  for (i, j) in enumerate(G)
    d = m.s * (u[j] + _linear(m.b, j, T))
    v[i] = d
    t += d * d
  end
  !isactive(x, k) && sqrt(t) <= c && return zero(T)

  # v = x[G] - ∇_G f(x) / L
  L = _groupcurvature!(m, x, k)
  t = zero(T)
  for (i, j) in enumerate(G)
    d = iszero(L) ? x[j] : x[j] - v[i] / L
    v[i] = d
    t += d * d
  end

  shrink = _groupshrink(sqrt(t), c, L)
  maxh = zero(T)
  for (i, j) in enumerate(G)
    new = shrink * v[i]
    h = new - x[j]
    iszero(h) && continue
    x[j] = new
    @inbounds @simd for l in eachindex(u)
      u[l] += A[l, j] * h
    end
    maxh = max(maxh, _scaledchange(m, x, j, h, scaled))
  end
  maxh
end
