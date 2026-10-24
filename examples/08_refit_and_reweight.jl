# Estimators built from lasso solves:
#
# - the least squares refit on the selected support, which removes the
#   shrinkage of the lasso (`refit!`);
# - the adaptive lasso, a second lasso whose weights come from the first;
# - iteratively reweighted ℓ1, which alternates between the solution and the
#   weights until they settle.
#
# `reweight!(x, f, g, update)` runs the last two: it solves, calls
# `update(g, x, m)` for the next penalty and the relative change it makes, and
# solves again from the previous solution, until the change is below `tol`.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics, Printf
using ProximalSolvers: residuals

rng = Xoshiro(8)
Random.seed!(8)
n, p = 200, 400
X = randn(rng, n, p)
β_true = zeros(p)
β_true[1:6] = [2.0, -2.0, 1.5, -1.5, 1.0, -1.0]
y = X * β_true + randn(rng, n)
f = (1 / n) * LeastSquaresLoss(y, X)
λ = sqrt(2 * log(p) / n)

β = SparseIterate(p)
solve!(β, f, NormL1(λ), CDSolver())
lasso = Vector(β);

# --- Refitting on the support ---
#
# `refit!(x, f)` minimizes the loss over the vectors with the support of x, by
# a pivoted QR factorization of the selected columns for least squares. The
# coefficients are no longer shrunk towards zero, and the error drops:

refit!(β, f)
refitted = Vector(β)
(lasso_error = norm(lasso - β_true), refit_error = norm(refitted - β_true),
 support = findall(!iszero, refitted))

# --- The adaptive lasso ---
#
# The adaptive lasso (Zou, 2006) penalizes coefficient k by λ/|β̃ₖ|, where β̃ is
# a first estimate, here the lasso: large coefficients are penalized little,
# and coordinates at zero in β̃ get an infinite weight and stay at zero. Two
# solves with one weight update are `maxiter = 2, tol = 0`.

adaptive_weights(g, x, m) = (NormL1(g.λ0, 1 ./ abs.(Vector(x))), Inf)
βa = SparseIterate(p)
ra = reweight!(βa, f, NormL1(λ), adaptive_weights; maxiter = 2, tol = 0)
adaptive = Vector(βa)
(solves = ra.iterations, error = norm(adaptive - β_true), support = findall(!iszero, adaptive))

# --- Iteratively reweighted ℓ1 ---
#
# The concave penalty λε Σₖ log(1 + |βₖ|/ε) matches the lasso near zero and
# grows only logarithmically, so it shrinks large coefficients less. Its
# tangent at the current β is a weighted ℓ1 norm with weights
# wₖ = 1/(1 + |βₖ|/ε), so solving the weighted lasso, updating the weights and
# solving again decreases the objective at every step (Candès, Wakin and Boyd,
# 2008). It starts from the lasso, all wₖ = 1, and stops when no weight changes
# by more than 1e-6.

ε = 0.1
function tangent_weights(g, x, m)
    w = 1 ./ (1 .+ abs.(Vector(x)) ./ ε)
    (NormL1(g.λ0, w), norm(w - g.w, Inf))
end
βr = SparseIterate(p)
rr = reweight!(βr, f, NormL1(λ, ones(p)), tangent_weights, CDSolver(; tol = 1e-10); tol = 1e-6, maxiter = 100)
reweighted = Vector(βr)
(solves = rr.iterations, error = norm(reweighted - β_true), support = findall(!iszero, reweighted))

# --- Checks ---
#
# The refit solves the normal equations on the support. The adaptive lasso
# satisfies the optimality conditions of its weighted lasso, and the reweighted
# solution those of the log penalty: ∇ₖf(β) = -λ sign(βₖ)/(1 + |βₖ|/ε) where
# βₖ ≠ 0, and |∇ₖf(β)| ≤ λ where βₖ = 0.

S = findall(!iszero, refitted)
@assert refitted[S] ≈ X[:, S] \ y
@assert norm(refitted - β_true) < norm(lasso - β_true)
function kkt(X, y, b, λw)
    grad = X' * (X * b - y) / size(X, 1)
    maximum(k -> iszero(b[k]) ? max(abs(grad[k]) - λw[k], 0.0) : abs(grad[k] + λw[k] * sign(b[k])), eachindex(b))
end
wa = ra.penalty.λ0 .* ra.penalty.w
@assert kkt(X, y, adaptive, wa) < 1e-6
@assert issubset(findall(!iszero, adaptive), findall(!iszero, lasso))
@assert issubset(1:6, findall(!iszero, adaptive))
@assert rr.converged && kkt(X, y, reweighted, λ ./ (1 .+ abs.(reweighted) ./ ε)) < 1e-6
@assert norm(reweighted - β_true) < norm(lasso - β_true)
