# The lasso by coordinate descent, in the scaling of glmnet:
#
#     minimize  ‖y - Xβ‖² / (2n) + λ ‖β‖₁
#
# `CDSolver` minimizes over one coordinate at a time. For least squares it keeps
# the residuals r = y - Xβ up to date, so the step of coordinate k, the soft
# thresholding of βₖ + Xₖ'r / ‖Xₖ‖² at nλ / ‖Xₖ‖², costs O(n). The iterate is a
# `SparseIterate`: it stores the coordinates that have been nonzero, which are
# the active set over which most passes go.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics

rng = Xoshiro(1)
Random.seed!(1)        # coordinate descent draws its random orders from the global generator
n, p = 200, 1000
X = randn(rng, n, p)
β_true = zeros(p)
β_true[1:5] = [2.0, -1.5, 1.0, -1.0, 0.5]
y = X * β_true + randn(rng, n)

f = (1 / n) * LeastSquaresLoss(y, X);       # ‖y - Xβ‖² / (2n)

# --- The penalty level ---
#
# Zero minimizes f + λ‖β‖₁ exactly when λ ≥ λmax = ‖∇f(0)‖∞ = ‖X'y‖∞ / n.
# `lambda_max` computes it for any loss and penalty that coordinate descent
# knows.

λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))
λ = 0.1 * λmax
β = SparseIterate(p)
res = solve!(β, f, NormL1(λ), CDSolver())

# The solver stops after a full pass in which no coordinate moved by more than
# `tol` (1e-7 by default), each change measured on the scale of the data:
# |Δβₖ| ‖Xₖ‖ / √n against `tol` times ‖y‖ / √n. The support:

findall(!iszero, β)

# --- The optimality conditions ---
#
# β minimizes f + λ‖β‖₁ if and only if the gradient ∇f(β) = X'(Xβ - y) / n
# satisfies ∇ₖf(β) = -λ sign(βₖ) where βₖ ≠ 0, and |∇ₖf(β)| ≤ λ where βₖ = 0.
# The largest violation:

b = Vector(β)
grad = X' * (X * b - y) / n
kkt = maximum(k -> iszero(b[k]) ? max(abs(grad[k]) - λ, 0.0) : abs(grad[k] + λ * sign(b[k])), 1:p)

# --- With an intercept ---
#
# `cdmodel(f, x; intercept = true)` minimizes ‖y - β₀ - Xβ‖² / (2n) over an
# unpenalized intercept β₀ as well. For a given β the best intercept is
# β₀ = ȳ - x̄'β, with the means ȳ of y and x̄ of the columns of X, and
# substituting it centers y and the columns. The model centers them on the fly,
# without copying X. A model, built once, serves any number of solves.

y3 = y .+ 3
β3 = SparseIterate(p)
m = cdmodel((1 / n) * LeastSquaresLoss(y3, X), β3; intercept = true)
solve!(β3, m, NormL1(λ), CDSolver())
β₀ = ProximalSolvers.intercept(m, β3)

# The slopes are those of the lasso on the centered data, and the intercept
# absorbs the shift of y:

Xc = X .- mean(X; dims = 1)
yc = y3 .- mean(y3)
βc = SparseIterate(p)
solve!(βc, (1 / n) * LeastSquaresLoss(yc, Xc), NormL1(λ), CDSolver(; tol = 1e-10))
(maximum(abs, Vector(β3) - Vector(βc)), β₀)

# --- Checks ---

@assert λmax ≈ norm(X' * y, Inf) / n
@assert res.converged
@assert kkt < 1e-5 * λ
@assert issubset(1:5, findall(!iszero, β))
@assert maximum(abs, Vector(β3) - Vector(βc)) < 1e-5
@assert β₀ ≈ mean(y3) - dot(mean(X; dims = 1), Vector(β3))
