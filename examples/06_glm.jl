# Penalized generalized linear models,
#
#     minimize  (1/n) Σᵢ ℓ(xᵢ'β + oᵢ, yᵢ) + λ ‖β‖₁,
#
# for the logistic, Poisson and Huber losses ℓ of ProximalBase. Two solvers:
#
# - `CDSolver` takes coordinate steps with a bound on the curvature of ℓ
#   (1/4 for the logistic loss, 1 for Huber's), which majorize the loss along
#   each coordinate. It needs ℓ'' to be bounded, so not the Poisson loss.
# - `ProxNewton` minimizes the quadratic model of the loss, a weighted least
#   squares problem, plus the penalty by coordinate descent, then backtracks on
#   the objective. This is how glmnet fits these models.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics, Printf

logistic(t) = 1 / (1 + exp(-t));

# --- Sparse logistic regression ---
#
# An unpenalized intercept is a column of ones with weight zero in the penalty.

rng = Xoshiro(6)
n, p = 500, 1000
X = [ones(n) randn(rng, n, p)]
β_true = zeros(p + 1)
β_true[1:6] = [-0.5, 1.5, -1.5, 1.0, -1.0, 0.8]
y = Float64.(rand(rng, n) .< logistic.(X * β_true))

f = (1 / n) * LogisticLoss(y, X)
w = [0.0; ones(p)]
λ = 0.2 * lambda_max(f, NormL1(1.0, w), SparseIterate(p + 1))
g = NormL1(λ, w)

fits = map(["coordinate descent" => CDSolver(; tol = 1e-10), "proximal Newton" => ProxNewton()]) do (name, solver)
    local β, res
    for _ in 1:2                       # the second run is timed: the first compiles
        Random.seed!(1)
        β = SparseIterate(p + 1)
        res = solve!(β, f, g, solver)
    end
    Random.seed!(1)
    t = @elapsed solve!(SparseIterate(p + 1), f, g, solver)
    @printf "%-20s %5d passes   %6.1f ms   objective %.10f\n" name res.passes 1000t res.minimum
    Vector(β)
end;

# The two solutions agree. Proximal Newton takes more passes in all, but its
# passes are over a weighted least squares model, whose coordinate steps cost
# O(n) multiplications; a coordinate step on the logistic loss itself evaluates
# ℓ′, an exponential, at every observation.

βcd, βpn = fits
findall(!iszero, βpn)

# --- Poisson regression with an exposure ---
#
# Counts yᵢ observed over exposures tᵢ, with mean tᵢ exp(xᵢ'β): the offset
# oᵢ = log tᵢ enters the linear predictor with coefficient one. The curvature
# of the Poisson loss, exp(η), is unbounded, so only `ProxNewton` applies.

# A Poisson draw with mean μ is the number of arrivals before time μ of a
# Poisson process of rate one, whose gaps are exponential.

function rand_poisson(rng, μ)
    k, s = 0, randexp(rng)
    while s < μ
        k += 1
        s += randexp(rng)
    end
    k
end

t = rand(rng, n) .* 4 .+ 1
Z = randn(rng, n, 200)
θ_true = zeros(200)
θ_true[1:3] = [0.5, -0.4, 0.3]
counts = Float64.(rand_poisson.(rng, t .* exp.(Z * θ_true)));

# `lambda_max` needs a loss that coordinate descent takes, so here λmax comes
# from the gradient at zero, (1/n) Z'(t - y), directly.

fp = (1 / n) * PoissonLoss(counts, Z; offset = log.(t))
λp = 0.2 * norm(Z' * (t .- counts), Inf) / n
θ = SparseIterate(200)
rp = solve!(θ, fp, NormL1(λp), ProxNewton())
(passes = rp.passes, support = findall(!iszero, θ), estimate = round.(Vector(θ)[1:3]; digits = 2))

# --- Huber regression ---
#
# With gross outliers in y, the Huber loss, quadratic for residuals below δ and
# linear beyond, keeps them from pulling the fit, where least squares does not.

yh = Z * θ_true + 0.5 * randn(rng, n)
yh[1:25] .+= 20                                   # 5% gross outliers
fh = (1 / n) * HuberLoss(yh, Z, 1.0)
fl = (1 / n) * LeastSquaresLoss(yh, Z)
θh, θl = SparseIterate(200), SparseIterate(200)
solve!(θh, fh, NormL1(0.05), CDSolver(; tol = 1e-10))
solve!(θl, fl, NormL1(0.05), CDSolver(; tol = 1e-10))
(huber_error = norm(Vector(θh) - θ_true), least_squares_error = norm(Vector(θl) - θ_true))

# --- Checks ---
#
# The optimality conditions, with the gradient (1/n) Z'ℓ'(η) of each loss.

function kkt(grad, b, λ, w = ones(length(b)))
    maximum(k -> iszero(b[k]) ? max(abs(grad[k]) - λ * w[k], 0.0) : abs(grad[k] + λ * w[k] * sign(b[k])), eachindex(b))
end
grad_logistic(b) = X' * (logistic.(X * b) .- y) / n
grad_poisson(b) = Z' * (t .* exp.(Z * b) .- counts) / n
grad_huber(b) = Z' * clamp.(Z * b .- yh, -1.0, 1.0) / n
@assert kkt(grad_logistic(βpn), βpn, λ, w) < 1e-7 && kkt(grad_logistic(βcd), βcd, λ, w) < 1e-7
@assert norm(βcd - βpn, Inf) < 1e-5
@assert issubset(2:6, findall(!iszero, βpn))
@assert rp.converged && kkt(grad_poisson(Vector(θ)), Vector(θ), λp) < 1e-7
@assert issubset(1:3, findall(!iszero, θ))
@assert kkt(grad_huber(Vector(θh)), Vector(θh), 0.05) < 1e-7
@assert norm(Vector(θh) - θ_true) < norm(Vector(θl) - θ_true)
