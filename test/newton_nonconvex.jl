# Proximal Newton with MCP and SCAD, whose coordinate problems are not convex
# where the curvature of the quadratic model is below their concavity (as for a
# logistic loss with a = 3): the paths reach stationary points, checked against
# the conditions computed from dense algebra.

module NewtonNonconvexTests

using Test, LinearAlgebra, Random, Statistics
using ProximalBase
using ProximalSolvers

logistic(z) = 1 / (1 + exp(-z))
dmcp(λ, a) = t -> sign(t) * max(λ - abs(t) / a, 0.0)
dscad(λ, a) = t -> sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))

# the largest violation of the stationarity conditions at b for the gradient G
stationarity(G, b, λ, d) = maximum(j -> b[j] == 0 ? max(abs(G[j]) - λ, 0.0) : abs(G[j] + d(b[j])), eachindex(b))

# AR(ρ) standardized columns, an intercept and six nonzero coefficients
function glmdata(family; n = 400, p = 40, ρ = 0.5, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
  end
  X = (X .- mean(X; dims = 1)) ./ std(X; dims = 1, corrected = false)
  β = zeros(p)
  β[1:6] .= family === :logistic ? [1.0, -0.8, 0.6, -0.6, 0.4, 0.3] : [0.4, -0.3, 0.3, -0.2, 0.2, 0.1]
  η = 0.2 .+ X * β
  y = family === :logistic ? Float64.(rand(rng, n) .< logistic.(η)) :
      Float64.([sum(cumsum(-log.(rand(rng, 60))) .< exp(e)) for e in η])
  X, y
end

newton = ProxNewton(; tol = 1e-14, maxiter = 200, inner = CDSolver(; tol = 1e-12))

@testset "MCP and SCAD paths: $family" for family in (:logistic, :poisson), seed in (1, 2)
  X, y = glmdata(family; seed)
  n, p = size(X)
  f = (1 / n) * (family === :logistic ? LogisticLoss(y, X) : PoissonLoss(y, X))
  m = ProximalSolvers.cdmodel(f, SparseIterate(p); intercept = true)
  mean_of(η) = family === :logistic ? logistic.(η) : exp.(η)
  for (g, d) in ((MCP(1.0, 3.0), dmcp), (SCAD(1.0, 3.7), dscad))
    λmax = lambda_max(m, g, SparseIterate(p))
    λs = λmax * (1 + 1e-9) .* 10 .^ range(0, -1.3; length = 20)
    β0s = Float64[]
    path = solve_path(m, g, λs, newton; x = SparseIterate(p), devmax = 1.0,
                      callback = (k, λ, x, m) -> (push!(β0s, ProximalSolvers.intercept(m, x)); false))
    @test all(path.converged)
    @test nnz(path.solutions[1]) == 0 && nnz(path.solutions[end]) > 0
    for k in eachindex(λs)
      b = Vector(path.solutions[k])
      G = X' * (mean_of(β0s[k] .+ X * b) .- y) / n
      @test stationarity(G, b, λs[k], d(λs[k], g.a)) < 1e-8
    end
  end
end

@testset "a SCAD path where the undamped steps stall" begin
  # columns on different scales, standardized for the penalty, as
  # CoordinateDescent's glmLasso does with standardizeX; without the damped
  # retry the step at the second level jumps toward a minimum of the model at
  # zero, the line search keeps a sliver of it, and the path stops with a
  # stationarity violation of 3e-4 (on Julia 1.13; the data depend on randn)
  rng = Xoshiro(501)
  n, p, ρ = 400, 40, 0.5
  X = randn(rng, n, p)
  for j in 2:p
    X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
  end
  c = 0.5 .+ 2 .* rand(rng, p)
  X .*= transpose(c)
  β = zeros(p); β[1:6] .= [1.0, -0.8, 0.6, -0.6, 0.4, 0.3] ./ c[1:6]
  y = Float64.(rand(rng, n) .< logistic.(0.2 .+ X * β))
  Xs = X ./ transpose(vec(sqrt.(mean((X .- mean(X; dims = 1)) .^ 2; dims = 1))))
  f = (1 / n) * LogisticLoss(y, Xs)
  m = ProximalSolvers.cdmodel(f, SparseIterate(p); intercept = true)
  g = SCAD(1.0, 3.7)
  λmax = lambda_max(m, g, SparseIterate(p))
  λs = λmax .* 0.05 .^ ((0:24) ./ 24)
  β0s = Float64[]
  path = solve_path(m, g, λs, newton; x = SparseIterate(p), devmax = 1.0,
                    callback = (k, λ, x, m) -> (push!(β0s, ProximalSolvers.intercept(m, x)); false))
  viol = map(eachindex(λs)) do k
    b = Vector(path.solutions[k])
    G = Xs' * (logistic.(β0s[k] .+ Xs * b) .- y) / n
    stationarity(G, b, λs[k], dscad(λs[k], 3.7))
  end
  @test maximum(viol) < 1e-6
end

@testset "a single level from zero, and on a box" begin
  X, y = glmdata(:logistic; seed = 3)
  n, p = size(X)
  f = (1 / n) * LogisticLoss(y, X)
  for (g, d) in ((MCP(0.05, 3.0), dmcp(0.05, 3.0)), (SCAD(0.05, 3.7), dscad(0.05, 3.7)))
    m = ProximalSolvers.cdmodel(f, SparseIterate(p); intercept = true)
    x = SparseIterate(p)
    r = solve!(x, m, g, newton)
    @test r.converged
    b = Vector(x)
    G = X' * (logistic.(ProximalSolvers.intercept(m, x) .+ X * b) .- y) / n
    @test stationarity(G, b, 0.05, d) < 1e-8
  end
  # on the box 0 ≤ x ≤ 0.3: the stationarity conditions with the normal cone
  m = ProximalSolvers.cdmodel(f, SparseIterate(p); intercept = true)
  x = SparseIterate(p)
  r = solve!(x, m, Constrained(MCP(0.02, 3.0), IndBox(0.0, 0.3)), newton)
  @test r.converged
  b = Vector(x)
  @test all(0 .<= b .<= 0.3)
  G = X' * (logistic.(ProximalSolvers.intercept(m, x) .+ X * b) .- y) / n
  d = dmcp(0.02, 3.0)
  viol = maximum(eachindex(b)) do j
    lo, hi = b[j] == 0 ? (-0.02, 0.02) : (d(b[j]), d(b[j]))
    lo = b[j] == 0.0 ? -Inf : lo           # the lower bound 0 adds (-∞, 0] to -G
    hi = b[j] == 0.3 ? Inf : hi            # the upper bound adds [0, ∞)
    t = -G[j]
    t < lo ? lo - t : t > hi ? t - hi : 0.0
  end
  @test viol < 1e-8
end

end
