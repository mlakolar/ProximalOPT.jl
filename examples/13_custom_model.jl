# Coordinate descent on a loss of your own. A loss is a subtype of `CDModel`
# that holds a cache, and implements a few functions of `ProximalSolvers`:
#
# - `initialize!(m, x)` makes the cache current for x;
# - `coordinate_derivatives(m, x, k)` returns the derivative G of the loss
#   along coordinate k and its curvature a (a bound on it, for a loss that is
#   not quadratic);
# - `update_cache!(m, x, k, h)` updates the cache after coordinate k moved by h.
#
# From these coordinate descent takes the step xₖ ← prox_{gₖ/a}(xₖ - G/a) for
# any separable penalty g. The scales of the stopping rule, `loss_value`, and
# `gradient!` (for `lambda_max`, screening and paths) are optional.
#
# Here the loss is least squares, s ‖y - Xβ‖² / 2, with a sparse design matrix
# X. The built-in model reads X by its entries, which costs a search for each
# entry of a `SparseMatrixCSC`; this one visits the stored entries of a column
# only, so a coordinate step costs O(nnz(X[:, k])).

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, SparseArrays, Printf

mutable struct SparseLeastSquares <: CDModel{Float64}
    X::SparseMatrixCSC{Float64, Int}
    y::Vector{Float64}
    s::Float64
    r::Vector{Float64}           # the residuals y - Xβ
    a::Vector{Float64}           # the curvatures s ‖X[:, k]‖²
end
SparseLeastSquares(X, y, s) = SparseLeastSquares(X, y, s, copy(y), s .* vec(sum(abs2, X; dims = 1)))

function ProximalSolvers.initialize!(m::SparseLeastSquares, β)
    rows, vals = rowvals(m.X), nonzeros(m.X)
    m.r .= m.y
    for k in activecoordinates(β), i in nzrange(m.X, k)
        m.r[rows[i]] -= vals[i] * β[k]
    end
    m
end

function ProximalSolvers.coordinate_derivatives(m::SparseLeastSquares, β, k::Int)
    rows, vals = rowvals(m.X), nonzeros(m.X)
    d = 0.0
    for i in nzrange(m.X, k)
        d += vals[i] * m.r[rows[i]]
    end
    (-m.s * d, m.a[k])                            # ∂ₖf = -s X[:, k]'r, curvature s ‖X[:, k]‖²
end

function ProximalSolvers.update_cache!(m::SparseLeastSquares, β, k::Int, h)
    rows, vals = rowvals(m.X), nonzeros(m.X)
    for i in nzrange(m.X, k)
        m.r[rows[i]] -= vals[i] * h
    end
    m
end

# The optional functions: the scales of the relative stopping rule (as for the
# built-in least squares, √aₖ and √s ‖y‖), the value of the loss, and its
# gradient -s X'r.

ProximalSolvers.coordinate_scale(m::SparseLeastSquares, β, k::Int) = sqrt(m.a[k])
ProximalSolvers.response_scale(m::SparseLeastSquares, β) = sqrt(m.s) * norm(m.y)
ProximalSolvers.loss_value(m::SparseLeastSquares, β) = m.s * sum(abs2, m.r) / 2
ProximalBase.gradient!(G, m::SparseLeastSquares, β) = (mul!(G, m.X', m.r); G .*= -m.s);

# --- A sparse lasso ---

rng = Xoshiro(13)
Random.seed!(13)
n, p = 2000, 5000
rows = rand(rng, 1:n, 10p)                        # 10 nonzeros in each column
X = sparse(rows, repeat(1:p; inner = 10), randn(rng, 10p), n, p)
β_true = zeros(p)
β_true[1:10] .= 2.0
y = X * β_true + 0.5 * randn(rng, n)

m = SparseLeastSquares(X, y, 1 / n)
λmax = lambda_max(m, NormL1(1.0), SparseIterate(p))
β = SparseIterate(p)
res = solve!(β, m, NormL1(0.05λmax), CDSolver())
(passes = res.passes, nonzeros = count(!iszero, β), found = issubset(1:10, findall(!iszero, β)))

# The same problem with the built-in model and a dense copy of X gives the
# same solution, at a cost of O(n) per coordinate step instead of O(nnz):

Xd = Matrix(X)
βd = SparseIterate(p)
solve!(βd, (1 / n) * LeastSquaresLoss(y, Xd), NormL1(0.05λmax), CDSolver())
solve!(SparseIterate(p), m, NormL1(0.05λmax), CDSolver())        # compile both before timing
t_sparse = @elapsed solve!(SparseIterate(p), m, NormL1(0.05λmax), CDSolver())
t_dense = @elapsed solve!(SparseIterate(p), (1 / n) * LeastSquaresLoss(y, Xd), NormL1(0.05λmax), CDSolver())
(difference = norm(Vector(β) - Vector(βd), Inf), seconds_sparse = t_sparse, seconds_dense = t_dense)

# --- Paths and the other drivers ---
#
# A model works with every driver of coordinate descent: `solve_path` (given
# the iterate to start from, since a model need not have columns), `reweight!`
# and, through `coordinate_derivatives` and `update_cache!`, `refit!`, which
# for a quadratic loss builds the Hessian on the support from them.

λs = λmax .* 10 .^ range(0, -1.3; length = 20)
path = solve_path(m, NormL1(1.0), λs; x = SparseIterate(p))
[count(!iszero, b) for b in path.solutions]'

# --- Checks ---

function kkt(b, λ)
    grad = X' * (X * b - y) / n
    maximum(k -> iszero(b[k]) ? max(abs(grad[k]) - λ, 0.0) : abs(grad[k] + λ * sign(b[k])), eachindex(b))
end
@assert λmax ≈ norm(X' * y, Inf) / n
@assert res.converged && kkt(Vector(β), 0.05λmax) < 1e-6 * λmax
@assert norm(Vector(β) - Vector(βd), Inf) < 1e-5
@assert all(kkt(Vector(path.solutions[k]), λs[k]) < 1e-6 * λmax for k in eachindex(λs))
@assert issubset(1:10, findall(!iszero, β))
