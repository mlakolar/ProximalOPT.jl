module ADMMTest

using Test
using Random
using LinearAlgebra
using ProximalBase
using ProximalOPT

Random.seed!(1)

@testset "ADMM" begin

  opt = ProximalOptions(; maxiter = 5000)
  p = 10
  Z = randn(50, p)
  S = Symmetric(Z'Z / 50)

  @testset "Gaussian likelihood + no penalty" begin
    X = Matrix(1.0I, p, p)
    res = solve!(X, ProxGaussLikelihood(S), ProxZero(), ADMMSolver(), opt)
    @test res.converged
    @test X ≈ inv(S) rtol = 1e-3
  end

  @testset "graphical lasso" begin
    λ = 0.1
    for solver in (ADMMSolver(), ADMMSolver(; ρ = 5., α = 1.6))
      X = Matrix(1.0I, p, p)
      res = solve!(X, ProxGaussLikelihood(S), ProxL1(λ), solver, opt)
      @test res.converged
      @test isposdef(Symmetric(X))

      # KKT conditions for tr(SX) - logdet(X) + λ‖X‖₁:
      # (S - X⁻¹)ᵢⱼ = -λ sign(Xᵢⱼ) where Xᵢⱼ ≠ 0, |(S - X⁻¹)ᵢⱼ| ≤ λ otherwise
      G = S - inv(X)
      nz = abs.(X) .> 1e-4
      @test G[nz] ≈ -λ * sign.(X[nz]) atol = 1e-2
      @test all(abs.(G[.!nz]) .<= λ + 1e-2)
    end
  end

  @testset "argument checks" begin
    @test_throws ErrorException ADMMSolver(; ρ = 0.)
    @test_throws ErrorException ADMMSolver(; α = 2.)
  end

end

end
