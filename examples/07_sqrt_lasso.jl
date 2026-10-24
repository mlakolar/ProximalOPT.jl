# The square-root lasso (Belloni, Chernozhukov and Wang, 2011),
#
#     minimize  ‖y - Xβ‖ / √n + λ ‖β‖₁,
#
# and the scaled lasso (Sun and Zhang, 2012), which estimates the noise level
# σ with β. The lasso needs λ proportional to σ, which is unknown; the
# square-root lasso takes λ = λ₀ ≈ √(2 log(p) / n) whatever σ is, because the
# gradient of ‖y - Xβ‖ / √n at the true β, -X'ε / (√n ‖ε‖), does not depend
# on the scale of the noise ε.
#
# `SqrtLeastSquaresLoss(y, X)` is ‖y - Xβ‖, without the 1/√n, so its penalty
# level is √n λ₀. The loss is not differentiable where the residuals vanish,
# and coordinate descent minimizes it exactly over each coordinate.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics, Printf
using ProximalSolvers: residuals

rng = Xoshiro(7)
Random.seed!(7)
n, p = 200, 500
X = randn(rng, n, p)
β_true = zeros(p)
β_true[1:5] = [1.5, -1.5, 1.0, -1.0, 1.0]
λ₀ = sqrt(2 * log(p) / n)

# --- One penalty level for any noise level ---

for σ in (0.5, 1.0, 2.0)
    y = X * β_true + σ * randn(rng, n)
    β = SparseIterate(p)
    solve!(β, SqrtLeastSquaresLoss(y, X), NormL1(sqrt(n) * λ₀), CDSolver())
    σ̂ = norm(y - X * Vector(β)) / sqrt(n)
    @printf "σ = %4.2f   σ̂ = %.3f   support %s\n" σ σ̂ findall(!iszero, β)
end

# With one λ₀ for the three noise levels, the supports stay close to the true
# one, 1:5, and ‖y - Xβ̂‖ / √n tracks σ, somewhat above it because the penalty
# shrinks the coefficients. The lasso would need a λ proportional to σ.

# --- The scaled lasso by reweighting ---
#
# The scaled lasso minimizes ‖y - Xβ‖² / (2nσ) + σ/2 + λ₀‖β‖₁ over β and σ.
# For a fixed σ it is the lasso with level λ₀σ, and for a fixed β the minimizer
# is σ = ‖y - Xβ‖ / √n. `reweight!` alternates the two: `update(g, x, m)`
# returns the next penalty and the relative change of σ, from the residuals
# that the model `m` keeps.

y = X * β_true + randn(rng, n)
f = (1 / n) * LeastSquaresLoss(y, X)
function update(g, x, m)
    σ̂ = norm(residuals(m)) / sqrt(n)
    σ_previous = g.λ0 / λ₀
    (NormL1(λ₀ * σ̂), abs(σ̂ - σ_previous) / σ_previous)
end
βs = SparseIterate(p)
scaled = reweight!(βs, f, NormL1(λ₀ * std(y)), update, CDSolver(; tol = 1e-12); tol = 1e-12, maxiter = 100)
(iterations = scaled.iterations, σ̂ = scaled.penalty.λ0 / λ₀)

# At the fixed point, X'(y - Xβ)/n ∈ λ₀σ̂ ∂‖β‖₁ with σ̂ = ‖y - Xβ‖/√n: the
# optimality condition of the square-root lasso. The two give one estimator:

βq = SparseIterate(p)
solve!(βq, SqrtLeastSquaresLoss(y, X), NormL1(sqrt(n) * λ₀), CDSolver(; tol = 1e-12))
norm(Vector(βq) - Vector(βs), Inf)

# --- From sufficient statistics ---
#
# `SqrtQuadraticFunction(A, b, c)` is √(x'Ax + 2b'x + c). With A = X'X/n,
# b = -X'y/n and c = y'y/n it is ‖y - Xx‖ / √n, and a coordinate step costs
# O(p) instead of O(n), with the same solution:

A, b, c = Symmetric(X' * X / n), -X' * y / n, dot(y, y) / n
βa = SparseIterate(p)
solve!(βa, SqrtQuadraticFunction(A, b, c), NormL1(λ₀), CDSolver(; tol = 1e-12))
norm(Vector(βa) - Vector(βq), Inf)

# --- A sparse precision matrix ---
#
# The same function regresses one variable on the others from their covariance
# matrix S: with A = S, b = -S[:, j] and c = S[j, j],
# √(x'Sx - 2S[:, j]'x + S[j, j]) is ‖Z[:, j] - Zx‖ / √n, and an infinite weight
# keeps xⱼ at zero. Sun and Zhang (2013) estimate the precision matrix Ω = Σ⁻¹
# column by column this way: Ωⱼⱼ = 1/σ̂ⱼ² and Ωₖⱼ = -β̂ₖ/σ̂ⱼ², from the
# scaled lasso of variable j, with the penalty weighted by √Sₖₖ. Of the two
# estimates of each Ωₖⱼ, the one smaller in magnitude is kept.

q, m = 30, 400
Ω = Matrix(SymTridiagonal(fill(1.0, q), fill(0.4, q - 1)))
Z = randn(rng, m, q) / cholesky(Ω).U'            # m draws from N(0, Ω⁻¹)
S = cov(Z; corrected = false)
λ₀q = sqrt(2 * log(q) / m)

Ω̂ = zeros(q, q)
for j in 1:q
    w = sqrt.(diag(S))
    w[j] = Inf
    fj = SqrtQuadraticFunction(S, -S[:, j], S[j, j]; check = false)
    βj = SparseIterate(q)
    solve!(βj, fj, NormL1(λ₀q, w), CDSolver(; tol = 1e-10))
    σ̂j = fj(Vector(βj))
    Ω̂[:, j] .= -Vector(βj) ./ σ̂j^2
    Ω̂[j, j] = 1 / σ̂j^2
end
Ω̂ = [abs(Ω̂[k, j]) <= abs(Ω̂[j, k]) ? Ω̂[k, j] : Ω̂[j, k] for k in 1:q, j in 1:q]
offdiag = [(k, j) for j in 1:q for k in 1:q if k != j]
true_edges = count(((k, j),) -> Ω[k, j] != 0, offdiag)
found = count(((k, j),) -> Ω[k, j] != 0 && Ω̂[k, j] != 0, offdiag)
false_edges = count(((k, j),) -> Ω[k, j] == 0 && Ω̂[k, j] != 0, offdiag)
(true_edges = true_edges, found = found, false_edges = false_edges,
 relative_error = opnorm(Ω̂ - Ω) / opnorm(Ω))

# --- Checks ---

@assert scaled.converged
@assert norm(Vector(βq) - Vector(βs), Inf) < 1e-8
@assert norm(Vector(βa) - Vector(βq), Inf) < 1e-8
r = y - X * Vector(βq)
grad = -X' * r / (sqrt(n) * norm(r))                  # the gradient of ‖y - Xβ‖ / √n
bq = Vector(βq)
@assert maximum(k -> iszero(bq[k]) ? max(abs(grad[k]) - λ₀, 0.0) : abs(grad[k] + λ₀ * sign(bq[k])), 1:p) < 1e-8
@assert issymmetric(Ω̂) && found == true_edges && false_edges <= 0.05 * length(offdiag)
