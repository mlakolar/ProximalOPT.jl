module GLMTests

using ProximalSolvers, ProximalBase, LinearAlgebra, Random, Statistics, Test
logistic(z) = 1 / (1 + exp(-z))
function kkt_l1(grad, x, λ)
  maximum(i -> x[i] == 0 ? max(abs(grad[i]) - λ, 0.0) : abs(grad[i] + λ * sign(x[i])), eachindex(x))
end
rng = Xoshiro(3)
n, p = 300, 500
X = randn(rng, n, p); β = zeros(p); β[1:5] .= [1.5, -1, 1, -0.8, 0.6]
y = Float64.(rand(rng, n) .< logistic.(X * β))
f = (1 / n) * LogisticLoss(y, X)
gradf(b) = X' * (logistic.(X * b) .- y) / n
λ = 0.1 * maximum(abs, gradf(zeros(p)))
@testset "generalized linear losses" begin
  x1 = SparseIterate(p)
  r1 = solve!(x1, f, NormL1(λ), CDSolver(; tol = 1e-10, maxpasses = 10^6))
  @test r1.converged
  @test kkt_l1(gradf(Vector(x1)), Vector(x1), λ) < 1e-6
  x2 = SparseIterate(p)
  r2 = solve!(x2, f, NormL1(λ), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-10)))
  @test r2.converged
  @test kkt_l1(gradf(Vector(x2)), Vector(x2), λ) < 1e-6
  @test Vector(x1) ≈ Vector(x2) atol = 1e-5
  # Poisson: unbounded curvature, ProxNewton only
  yp = Float64.([rand(rng) < 0.5 ? 0 : rand(rng, 0:4) for _ in 1:n])
  fp = (1 / n) * PoissonLoss(yp, X)
  @test_throws ArgumentError solve!(SparseIterate(p), fp, NormL1(0.1), CDSolver())
  gp(b) = X' * (exp.(X * b) .- yp) / n
  λp = 0.2 * maximum(abs, gp(zeros(p)))
  x3 = SparseIterate(p)
  r3 = solve!(x3, fp, NormL1(λp), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-10)))
  @test r3.converged
  @test kkt_l1(gp(Vector(x3)), Vector(x3), λp) < 1e-6
  # Poisson counts around 50: from zero, the first Newton steps overshoot, and
  # the line search shortens them
  ηtrue = 4 .+ 0.3 .* (X[:, 1:3] * [1.0, -1.0, 0.5])
  yc = Float64.(round.(exp.(ηtrue)))
  fc = (1 / n) * PoissonLoss(yc, X)
  gc(b) = X' * (exp.(X * b) .- yc) / n
  λc = 0.05 * maximum(abs, gc(zeros(p)))
  x5 = SparseIterate(p)
  r5 = solve!(x5, fc, NormL1(λc), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-10)))
  @test r5.converged
  @test kkt_l1(gc(Vector(x5)), Vector(x5), λc) < 1e-6 * maximum(abs, gc(zeros(p)))
  # warm starts: proximal Newton and coordinate descent with the curvature bound
  x6 = SparseIterate(Vector(x2) .+ 0.1 .* randn(rng, p))
  solve!(x6, f, NormL1(λ), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-10)))
  @test Vector(x6) ≈ Vector(x2) atol = 1e-6
  x7 = SparseIterate(Vector(x2) .+ 0.1 .* randn(rng, p))
  solve!(x7, f, NormL1(λ), CDSolver(; tol = 1e-10, maxpasses = 10^6))
  @test Vector(x7) ≈ Vector(x2) atol = 1e-5
  # an unscaled loss: the same solution at the penalty level scaled by n
  x8 = SparseIterate(p)
  solve!(x8, LogisticLoss(y, X), NormL1(n * λ), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-10)))
  @test Vector(x8) ≈ Vector(x2) atol = 1e-6
  # Huber
  yh = X * β + randn(rng, n) .* 2; yh[1:10] .+= 30
  fh = (1 / n) * HuberLoss(yh, X, 1.0)
  x4 = SparseIterate(p)
  r4 = solve!(x4, fh, NormL1(0.05), CDSolver(; tol = 1e-10, maxpasses = 10^6))
  @test r4.converged
  ψ(r) = clamp(r, -1.0, 1.0)
  @test kkt_l1(X' * ψ.(X * Vector(x4) - yh) / n, Vector(x4), 0.05) < 1e-6
end

end
