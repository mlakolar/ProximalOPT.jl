module LowRankTest

using Test
using Random
using LinearAlgebra
using ProximalBase
using ProximalSolvers

Random.seed!(1)

# multi-response regression Y = XB + E with a p × k coefficient matrix of rank r,
# the data of the README example: f(B) = ‖Y - XB‖²_F / 2 and g(B) = λ ‖B‖_*
n, p, k, r = 200, 20, 10, 2
X = randn(n, p)
B_true = randn(p, r) * randn(r, k)
Y = X * B_true + 0.5 * randn(n, k)
f = LeastSquaresLoss(Y, X)
∇f(B) = X' * (X * B - Y)

# for Gaussian noise, λ = σ‖X‖(√p + √k) bounds the operator norm of
# ∇f(B_true) = -X'E; σ is estimated from the residuals of least squares
σ = norm(Y - X * (X \ Y)) / sqrt((n - p) * k)
λ = σ * opnorm(X) * (sqrt(p) + sqrt(k))
g = NuclearNorm(λ)

opt = ProximalOptions(; ftol = 1e-12, xtol = 1e-12, gradtol = 1e-10, maxiter = 10000)

solvers = (ProximalGradientDescent(), AccProxGradDescent())

@testset "low-rank multi-response regression" begin

  Bs = map(solvers) do solver
    B = zeros(p, k)
    res = solve!(B, f, g, solver, opt)
    @testset "$(summary(solver))" begin
      @test res.minimizer === B
      @test ProximalSolvers.converged(res)
      @test res.minimum ≈ f(B) + g(B)

      # the rank of B_true is recovered
      @test rank(B; atol = 1e-6) == r
      @test norm(B - B_true) / norm(B_true) < 0.05

      # B is a fixed point of the proximal gradient step with step size 1/‖X‖²
      γ = 1 / opnorm(X)^2
      @test prox(g, B - γ * ∇f(B), γ) ≈ B rtol = 1e-6

      # -∇f(B) / λ is a subgradient of ‖⋅‖_* at B ≠ 0, so its operator norm is 1
      @test opnorm(∇f(B)) ≈ λ rtol = 1e-4
    end
    B
  end
  @test Bs[1] ≈ Bs[2] rtol = 1e-5

  @testset "zero solution" begin
    # B = 0 for λ ≥ ‖∇f(0)‖ = ‖X'Y‖; just below, B has rank one
    λmax = opnorm(X' * Y)
    for solver in solvers
      B = zeros(p, k)
      solve!(B, f, NuclearNorm(1.01 * λmax), solver, opt)
      @test B == zeros(p, k)

      B = zeros(p, k)
      solve!(B, f, NuclearNorm(0.99 * λmax), solver, opt)
      @test rank(B; atol = 1e-6) == 1
    end
  end

end

end
