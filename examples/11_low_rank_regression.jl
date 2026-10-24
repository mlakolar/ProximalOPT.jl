# Low-rank multi-response regression: k responses on the same p predictors,
# Y = XB + E, with a p × k coefficient matrix B of low rank, by
#
#     minimize  ‖Y - XB‖²_F / 2 + λ ‖B‖_*.
#
# The nuclear norm ‖B‖_*, the sum of the singular values of B, sets the small
# singular values to exactly zero, as the ℓ1 norm of the lasso does with small
# coefficients: its proximal operator soft-thresholds the singular values. The
# proximal gradient solvers take a matrix as the starting point, or an array of
# any shape that f and g accept.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random

rng = Xoshiro(11)
n, p, k = 200, 20, 10
X = randn(rng, n, p)
B_true = randn(rng, p, 2) * randn(rng, 2, k)           # rank 2
Y = X * B_true + 0.5 * randn(rng, n, k);

# --- The penalty level ---
#
# The gradient of f at B_true is -X'E. For Gaussian noise with standard
# deviation σ, its operator norm, the largest singular value, is at most about
# σ‖X‖(√p + √k), and a λ above it keeps the noise out of the estimate. σ comes
# from the residuals of least squares.

B_ls = X \ Y
σ = norm(Y - X * B_ls) / sqrt((n - p) * k)
λ = σ * opnorm(X) * (sqrt(p) + sqrt(k))

f = LeastSquaresLoss(Y, X)                              # ‖Y - XB‖²_F / 2
g = NuclearNorm(λ)                                      # λ ‖B‖_*
options = ProximalOptions(; maxiter = 2000, ftol = 1e-12, gradtol = 1e-10)
B = zeros(p, k)
res = solve!(B, f, g, AccProxGradDescent(), options)
(rank = rank(B; atol = 1e-6), error = norm(B - B_true) / norm(B_true),
 least_squares_rank = rank(B_ls), least_squares_error = norm(B_ls - B_true) / norm(B_true))

# The estimate has the rank of B_true, while least squares has full rank. The
# penalty also shrinks the singular values that it keeps, by λ, so a smaller λ
# has less bias but keeps more singular values, and for λ ≥ ‖X'Y‖ the solution
# is zero.

for scale in (0.5, 0.25)
    Bs = zeros(p, k)
    solve!(Bs, f, NuclearNorm(scale * λ), AccProxGradDescent(), options)
    println("λ × ", scale, ": rank ", rank(Bs; atol = 1e-6), ", relative error ", round(norm(Bs - B_true) / norm(B_true); digits = 4))
end

# --- Checks ---
#
# At a minimizer, -∇f(B) = X'(Y - XB) is a subgradient of λ‖B‖_*: with the thin
# singular value decomposition B = UΣV' of rank r,
# X'(Y - XB) = λ(UV' + W), where U'W = 0, WV = 0 and ‖W‖ ≤ 1.

G = X' * (Y - X * B)
F = svd(B)
r = rank(B; atol = 1e-6)
U, V = F.U[:, 1:r], F.V[:, 1:r]
W = G / λ - U * V'
@assert res.f_converged || res.x_converged || res.grad_converged
@assert r == 2
@assert norm(U' * W) < 1e-5 && norm(W * V) < 1e-5 && opnorm(W) <= 1 + 1e-5
@assert iszero(solve!(zeros(p, k), f, NuclearNorm(1.01 * opnorm(X' * Y)), AccProxGradDescent()).minimizer)
