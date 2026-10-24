####################################
#
#  refits that reuse a factorization
#
####################################

"""
    RefitCache(T = Float64)

The Cholesky factor of the Hessian of a quadratic loss on a support, which
[`refit!`](@ref) keeps between calls with `refit!(x, f; support, cache)`. A
later refit of the same `f` on a support that differs by a few coordinates
updates the factor instead of computing it again:
- a coordinate that enters borders it, in O(n|S| + |S|²) for least squares;
- one that leaves is removed by Givens rotations, in O(|S|²).

So refits along a path of supports, as after a lasso path, cost about as much
as the gradient on the support, instead of O(n|S|²) each. When many coordinates
change, it is cheaper to factor again, which the cache then does.

It serves least squares (`LeastSquaresLoss`, a least-squares `GLMLoss`, either
scaled) and `QuadraticFunction`; for other losses the cache is ignored. When an
entering coordinate is nearly a combination of the others (a new pivot below
`√eps` of its curvature), the refit falls back to the pivoted QR factorization
that `refit!` uses without a cache, which also handles a rank-deficient
support, and the cache starts over.

The factor belongs to one loss: pass the same `f` (or the same model) to every
refit, don't change its data in between, and don't share the cache between
threads. `refactors` and `updates` count the factorizations from scratch and the
coordinates added or removed by updates.
"""
mutable struct RefitCache{T<:AbstractFloat}
  owner::Any            # the loss or model of the factor
  S::Vector{Int}        # the coordinates of the factor, in its order
  R::Matrix{T}          # R[1:k, 1:k] is upper triangular and R'R = H[S, S], k = length(S)
  h::Vector{T}          # a column of the Hessian, and solves
  refactors::Int
  updates::Int
end

RefitCache(::Type{T} = Float64) where {T<:AbstractFloat} = RefitCache{T}(nothing, Int[], zeros(T, 0, 0), T[], 0, 0)

# the models with the Hessian entries the cache needs
_has_hessian_columns(::CDModel) = false
_has_hessian_columns(::Union{LSModel, QuadModel}) = true

# h[q] = H[S[q], j]: s ∑ᵢ wᵢ (Xᵢₖ - x̄ₖ)(Xᵢⱼ - x̄ⱼ) for least squares (there is no
# intercept, so no centering, with weights) and s A[k, j] for a quadratic
function _hessian_column!(h, m::LSModel{T}, S, j::Int) where {T}
  X, c, w = m.X, m.xbar, m.w
  @inbounds for (q, k) in enumerate(S)
    acc = zero(T)
    if w === nothing
      ck, cj = c[k], c[j]
      @simd for i in axes(X, 1)
        acc += (X[i, k] - ck) * (X[i, j] - cj)
      end
    else
      @simd for i in axes(X, 1)
        acc += w[i] * X[i, k] * X[i, j]
      end
    end
    h[q] = m.s * acc
  end
  h
end
_hessian_diag(m::LSModel, j::Int) = m.s * _sqnorm!(m, j)

function _hessian_column!(h, m::QuadModel, S, j::Int)
  @inbounds for (q, k) in enumerate(S)
    h[q] = m.s * m.A[k, j]
  end
  h
end
_hessian_diag(m::QuadModel, j::Int) = m.s * m.A[j, j]

function _capacity!(cache::RefitCache{T}, k::Int) where {T}
  if size(cache.R, 1) < k
    n = max(k, 2 * size(cache.R, 1), 16)
    R = zeros(T, n, n)
    m = length(cache.S)
    R[1:m, 1:m] .= view(cache.R, 1:m, 1:m)
    cache.R = R
    resize!(cache.h, n)
  end
  cache
end

# border the factor with coordinate j: R'r = H[S, j], ρ² = H[j, j] - ‖r‖²; false
# when ρ² is too small, that is, j is nearly in the span of the others
function _add_coordinate!(cache::RefitCache{T}, m, j::Int) where {T}
  k = length(cache.S)
  _capacity!(cache, k + 1)
  R, r = cache.R, cache.h
  _hessian_column!(r, m, cache.S, j)
  @inbounds for q in 1:k
    t = r[q]
    for l in 1:q-1
      t -= R[l, q] * r[l]
    end
    r[q] = t / R[q, q]
  end
  c = _hessian_diag(m, j)
  ρ2 = c - sum(abs2, view(r, 1:k))
  (isfinite(ρ2) && ρ2 > sqrt(eps(T)) * c) || return false
  @inbounds for q in 1:k
    R[q, k+1] = r[q]
    R[k+1, q] = zero(T)
  end
  R[k+1, k+1] = sqrt(ρ2)
  push!(cache.S, j)
  true
end

# delete the coordinate in position q: drop column q of R, which leaves the
# columns after it upper Hessenberg, and restore the triangle by Givens
# rotations of the rows; R'R stays H on the remaining coordinates
function _remove_coordinate!(cache::RefitCache{T}, q::Int) where {T}
  k = length(cache.S)
  R = cache.R
  @inbounds for col in q:k-1, row in 1:col+1
    R[row, col] = R[row, col+1]
  end
  @inbounds for col in q:k-1
    a, b = R[col, col], R[col+1, col]
    ρ = hypot(a, b)
    c, s = a / ρ, b / ρ
    R[col, col], R[col+1, col] = ρ, zero(T)
    for j in col+1:k-1
      u, v = R[col, j], R[col+1, j]
      R[col, j] = c * u + s * v
      R[col+1, j] = c * v - s * u
    end
  end
  deleteat!(cache.S, q)
  cache
end

# bring the factor to the coordinates S; false if a new pivot was too small
function _update_factor!(cache::RefitCache, f, m, S::Vector{Int})
  if cache.owner !== f
    cache.owner = f
    empty!(cache.S)
  end
  inS, inF = Set(S), Set(cache.S)
  removed = [k for k in cache.S if !(k in inS)]
  added = [k for k in S if !(k in inF)]
  if length(removed) + length(added) > max(4, length(S) ÷ 4)
    empty!(cache.S)
    added = S
    cache.refactors += 1
  else
    for k in removed
      _remove_coordinate!(cache, findfirst(==(k), cache.S))
    end
    cache.updates += length(removed) + length(added)
  end
  for j in added
    if !_add_coordinate!(cache, m, j)
      empty!(cache.S)
      return false
    end
  end
  true
end

# the Newton step on the support from the cached factor: R'R δ = -G_S, solved
# in the order of the factor and returned in the order of S; nothing when the
# factor can't be used (another model, another element type, a small pivot), so
# that refit! falls back to its QR factorization
function _refit_step(cache::RefitCache{T}, f, m::CDModel{U}, x, S::Vector{Int}) where {T, U}
  (T === U && _has_hessian_columns(m)) || return nothing
  isempty(S) && return T[]
  _update_factor!(cache, f, m, S) || return nothing
  k = length(S)
  R, z = cache.R, cache.h
  @inbounds for (q, j) in enumerate(cache.S)
    z[q] = -coordinate_derivatives(m, x, j)[1]
  end
  @inbounds for q in 1:k                      # R'w = z
    t = z[q]
    for l in 1:q-1
      t -= R[l, q] * z[l]
    end
    z[q] = t / R[q, q]
  end
  @inbounds for q in k:-1:1                   # Rδ = w
    t = z[q]
    for l in q+1:k
      t -= R[q, l] * z[l]
    end
    z[q] = t / R[q, q]
  end
  position = Dict(j => q for (q, j) in enumerate(cache.S))
  [z[position[j]] for j in S]
end
