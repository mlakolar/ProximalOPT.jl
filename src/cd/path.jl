####################################
#
#  Penalty levels: λmax, paths, screening
#
####################################

# a zero iterate of the type and size of x
_zero_like(x::SparseIterate{T}) where {T} = SparseIterate(T, size(x)...)
_zero_like(x::SymmetricSparseIterate{T}) where {T} = SymmetricSparseIterate(T, x.p)
_zero_like(x::AtomIterate{T}) where {T} = AtomIterate(T, length(x), x.groups)

# the smallest level of g with a zero solution, from the gradient grad at zero:
# maxₖ ‖∇ₖ f(0)‖ / wₖ over the penalized coordinates; those with weight zero are
# not zero at any level
function _lambda_max(x, grad::AbstractArray{T}, g) where {T}
  λmax = zero(T)
  for k in 1:ncoordinates(x)
    w = _coordweight(g, x, k)
    iszero(w) && continue
    t = _gradnorm(grad, x, k) / w
    t > λmax && (λmax = t)
  end
  λmax
end

"""
    lambda_max(f, g, x)

The smallest penalty level `λ` for which zero minimizes `f + g` with the level
of `g` replaced by `λ` (`NormL1(λ, w)` for `g = NormL1(λ0, w)`, and so on):
`maxₖ |∇ₖf(0)| / wₖ` over the coordinates of iterates like `x` whose weight is
not zero, with `‖∇_G f(0)‖₂` for the groups of a `GroupNormL2`.
"""
function lambda_max(f::Union{AbstractFunction, CDModel}, g::AbstractFunction, x)
  m = f isa CDModel ? f : cdmodel(f, x)
  x0 = _zero_like(x)
  initialize!(m, x0)
  grad = _gradient_buffer(x0)
  gradient!(grad, m, x0)
  _lambda_max(x0, grad, g)
end

# Solves at the penalty g, warm started from x, with the sequential strong rule
# of Tibshirani et al. (2012) for the step from the level λprev to λ = level(g):
# full passes visit only the coordinates with ‖∇ₖf‖ ≥ (2λ - λprev) wₖ at the
# previous solution, the nonzero ones and the unpenalized ones. After
# convergence the optimality conditions ‖∇ₖf‖ ≤ λ wₖ of the other coordinates
# are checked with one full gradient, and violators are added and the problem
# solved again. m must be current for x, and grad holds the gradient at x on
# entry and on return; inS has one entry per coordinate. Returns the number of
# checks; st.passes holds the passes of all the solves.
function _screened!(x, m::CDModel{T}, g, λprev, grad::AbstractArray, st::_CDState{T}, s::CDSolver,
                    inS::BitVector) where {T}
  λ = _level(g)
  threshold = 2λ - λprev
  for k in eachindex(inS)
    w = _coordweight(g, x, k)
    inS[k] = iszero(w) || _isnonzero(x, k) || _gradnorm(grad, x, k) >= threshold * w
  end
  passes, checks = 0, 0
  while true
    st.candidates = findall(inS)
    _cd!(x, m, g, st, s)
    passes += st.passes
    gradient!(grad, m, x)
    checks += 1
    violated = false
    for k in eachindex(inS)
      if !inS[k] && _gradnorm(grad, x, k) > λ * _coordweight(g, x, k)
        inS[k] = true
        violated = true
      end
    end
    violated || break
  end
  st.candidates = nothing
  st.passes = passes
  checks
end

"""
    solve!(x, f, g, method::CDSolver; continuation)

With `continuation = n > 0`, `x` is set to zero and `f + g` is minimized along
`n` penalty levels spaced geometrically from [`lambda_max`](@ref) down to the
level of `g`, then at `g` itself, each warm started from the previous solution.
With `screening`, each level is screened with the strong rule (see
[`solve_path`](@ref)); the default screens except on a `SymmetricSparseIterate`,
whose coordinate steps update the gradient everywhere, so that the optimality
checks of screening cost more passes than they save. When the level of `g` is
zero or at least `lambda_max`, `f + g` is minimized from zero directly, and so
it is for a penalty without a level (`ZeroFunction`, `IndBox`, …), which has
no path.
"""
function solve!(x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate}, f::Union{AbstractFunction, CDModel},
                g::AbstractFunction, method::CDSolver; continuation::Integer,
                screening::Bool = !(x isa SymmetricSparseIterate))
  continuation >= 0 || throw(ArgumentError("continuation must be nonnegative, but it is $continuation."))
  continuation == 0 && return solve!(x, f, g, method)
  m = f isa CDModel ? f : cdmodel(f, x)
  _check_iterate(g, x)
  check_penalty(m, g)
  T = eltype(m)
  fill!(x, zero(T))
  initialize!(m, x)
  st = _CDState(T, x, method.random)
  passes, checks = 0, 0
  path = false
  if _haslevel(g)
    grad = _gradient_buffer(x)
    gradient!(grad, m, x)
    λmax = _lambda_max(x, grad, g)
    λ = _level(g)
    path = zero(λmax) < λ < λmax
  end
  if path
    l1, l2 = log(λmax), log(λ)
    inS = falses(ncoordinates(x))
    λprev = λmax
    for k in 0:continuation
      gk = k == continuation ? g : _with_level(g, exp(l1 + k * (l2 - l1) / continuation))
      if screening
        checks += _screened!(x, m, gk, λprev, grad, st, method, inS)
      else
        _cd!(x, m, gk, st, method)
      end
      passes += st.passes
      λprev = _level(gk)
    end
  else
    _cd!(x, m, g, st, method)
    passes += st.passes
  end
  _warn_not_converged(st, method)
  CDResults(x, loss_value(m, x) + g(x), passes, st.converged, st.maxchange, st.tolerance, checks)
end

Base.eltype(::CDModel{T}) where {T} = T

"""
    PathResults

Returned by [`solve_path`](@ref): the penalty levels `λ` that were solved
for, the `solutions` (a `SparseVector` per level for a vector iterate, a sparse
matrix for a matrix iterate), the objective `values`, and per level the
`passes`, whether coordinate descent `converged`, and the number of
`kkt_checks` of the screening.
"""
struct PathResults{S, T}
  λ::Vector{T}
  solutions::Vector{S}
  values::Vector{T}
  passes::Vector{Int}
  converged::Vector{Bool}
  kkt_checks::Vector{Int}
end

# a compact copy of the solution
_snapshot(x::SparseIterate{T, 1}) where {T} = _sparsevec(x)
_snapshot(x::AtomIterate) = sparsevec(Vector(x))
_snapshot(x::SymmetricSparseIterate) = sparse(Matrix(x))
_snapshot(x::SparseIterate{T, 2}) where {T} = sparse(Matrix(x))

function _sparsevec(x::SparseIterate{T, 1}) where {T}
  ind = Int[]
  val = T[]
  for k in activecoordinates(x)
    v = x[k]
    iszero(v) && continue
    push!(ind, k); push!(val, v)
  end
  p = sortperm(ind)
  sparsevec(ind[p], val[p], length(x))
end

"""
    solve_path(f, g, λs, method = CDSolver(); x, screening = true, maxnnz = Inf, callback = nothing)

Minimize `f + gₖ` along the decreasing penalty levels `λs`, where `gₖ` is `g`
with level `λs[k]`, each warm started from the previous solution. The iterate
`x` (a zero `SparseIterate` of the right length when `f` has columns, which
must otherwise be given) is the starting point and holds the last solution on
return.

With `screening`, each level is solved first over the coordinates kept by the
sequential strong rule of Tibshirani et al. (2012), and the optimality
conditions of the others are checked afterwards; violators are added and the
problem solved again. The path stops early after the first solution with more
than `maxnnz` nonzero coordinates, or when `callback(k, λ, x, m)` returns `true`
(`m` is the [`CDModel`](@ref), current for `x`; use it, for example, to compute
a validation loss). Returns [`PathResults`](@ref).
"""
function solve_path(f::Union{AbstractFunction, CDModel}, g::AbstractFunction, λs::AbstractVector{<:Real},
                    method::CDSolver = CDSolver(); x = _default_iterate(f), screening::Bool = true,
                    maxnnz::Real = Inf, callback = nothing)
  all(λ -> λ >= 0, λs) || throw(ArgumentError("The penalty levels must be nonnegative."))
  issorted(λs; rev = true) || throw(ArgumentError("The penalty levels must be decreasing."))
  m = f isa CDModel ? f : cdmodel(f, x)
  T = eltype(m)
  _check_iterate(g, x)
  check_penalty(m, g)
  isempty(λs) || _with_level(g, T(first(λs)))      # a penalty with a level
  initialize!(m, x)
  st = _CDState(T, x, method.random)
  if screening
    grad = _gradient_buffer(x)
    gradient!(grad, m, x)
    inS = falses(ncoordinates(x))
    λprev = _lambda_max(x, grad, g)      # λmax at the starting point, which need not be zero
  end
  solutions = Any[]
  out = PathResults(T[], [], T[], Int[], Bool[], Int[])
  for (k, λ) in enumerate(λs)
    gk = _with_level(g, T(λ))
    if screening
      initialize!(m, x)       # no drift of the cache along the path
      checks = _screened!(x, m, gk, λprev, grad, st, method, inS)
      λprev = T(λ)
    else
      _cd!(x, m, gk, st, method)
      checks = 0
    end
    _warn_not_converged(st, method)
    push!(out.λ, T(λ)); push!(solutions, _snapshot(x)); push!(out.values, loss_value(m, x) + gk(x))
    push!(out.passes, st.passes); push!(out.converged, st.converged); push!(out.kkt_checks, checks)
    (callback !== nothing && callback(k, T(λ), x, m) === true) && break
    nactive(x) > maxnnz && break
  end
  S = isempty(solutions) ? Any : typeof(first(solutions))
  PathResults{S, T}(out.λ, convert(Vector{S}, solutions), out.values, out.passes, out.converged, out.kkt_checks)
end

_default_iterate(f::LeastSquaresLoss{T, <:AbstractVector}) where {T} = SparseIterate(T, size(f.X, 2))
_default_iterate(f::Scaled) = _default_iterate(f.f)
_default_iterate(f::GLMLoss{L, T}) where {L, T} = SparseIterate(T, size(f.X, 2))
_default_iterate(f::QuadraticFunction{T}) where {T} = SparseIterate(T, size(f.A, 2))
_default_iterate(f::SqrtQuadraticFunction{T}) where {T} = SparseIterate(T, length(f.b))
_default_iterate(f) = throw(ArgumentError("give the starting iterate x of the path"))
