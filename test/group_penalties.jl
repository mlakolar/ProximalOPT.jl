# Block coordinate descent on an AtomIterate with penalties other than
# GroupNormL2: the group norms of ProximalBase through their own block step, and
# separable penalties entry by entry after the group's gradient step.

module GroupPenaltiesTests

using Test, LinearAlgebra, Random
using ProximalBase
using ProximalSolvers
const PS = ProximalSolvers

function groupdata(; n = 80, p = 60, seed = 21)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  β = zeros(p); β[1:4] .= [2, -1.5, 1, -1]; β[11:13] .= [1, 1, -1]
  X, X * β + 0.3 * randn(rng, n)
end

# a reference solution by accelerated proximal gradient to high accuracy
function reference(f, g, x0)
  res = solve!(copy(x0), f, g, AccProxGradDescent(), ProximalOptions(; maxiter = 50_000, ftol = 1e-15, xtol = 1e-14, gradtol = 1e-12))
  res.minimizer
end

@testset "GroupNormLinf" begin
  X, y = groupdata()
  n, p = size(X)
  groups = [i:i+4 for i in 1:5:p]
  λ = 0.1 * maximum(G -> norm(X[:, G]' * y, 1), groups) / n
  g = GroupNormLinf(λ, groups)
  for f in ((1 / n) * LeastSquaresLoss(y, X), QuadraticFunction(X' * X / n, -X' * y / n))
    for order in (:cyclic, :random)
      x = AtomIterate(Float64, p, groups)
      r = solve!(x, f, g, CDSolver(; order, tol = 1e-12))
      @test r.converged
      xref = reference(f, g, zeros(p))
      @test Vector(x) ≈ xref atol = 1e-6
      @test f(Vector(x)) + g(Vector(x)) ≈ f(xref) + g(xref) rtol = 1e-10
      # optimality of the zero groups: ‖∇_G f‖₁ ≤ λ wₖ
      grad = gradient(f, Vector(x))
      for (k, G) in enumerate(groups)
        all(iszero, Vector(x)[G]) && @test norm(grad[G], 1) <= λ + 1e-8
      end
    end
  end
  # a dense vector becomes an AtomIterate with the groups of g
  f = (1 / n) * LeastSquaresLoss(y, X)
  xd = zeros(p)
  solve!(xd, f, g, CDSolver(; tol = 1e-12))
  @test xd ≈ reference(f, g, zeros(p)) atol = 1e-6
  # the iterate must have the groups of g
  @test_throws ArgumentError solve!(SparseIterate(p), f, g, CDSolver())
  @test_throws ArgumentError solve!(AtomIterate(Float64, p, [1:p]), f, g, CDSolver())
end

@testset "separable penalties on an AtomIterate" begin
  X, y = groupdata(; seed = 22)
  n, p = size(X)
  groups = [i:i+2 for i in 1:3:p]
  f = (1 / n) * LeastSquaresLoss(y, X)
  λ = 0.1 * norm(X' * y, Inf) / n
  for g in (NormL1(λ), ElasticNet(λ, 0.1), IndBox(-0.5, 0.5))
    xa = AtomIterate(Float64, p, groups)
    solve!(xa, f, g, CDSolver(; tol = 1e-12))
    xs = SparseIterate(p)
    solve!(xs, f, g, CDSolver(; tol = 1e-12))
    @test Vector(xa) ≈ Vector(xs) atol = 1e-7
  end
end

@testset "a group that f does not depend on" begin
  X, y = groupdata(; seed = 23)
  n, p = size(X)
  X[:, 6:10] .= 0
  groups = [i:i+4 for i in 1:5:p]
  f = (1 / n) * LeastSquaresLoss(y, X)
  x = AtomIterate(ones(p), groups)
  solve!(x, f, GroupNormLinf(0.01, groups), CDSolver(; tol = 1e-10))
  @test all(iszero, Vector(x)[6:10])             # the minimizer of the penalty
  x = AtomIterate(ones(p), groups)
  solve!(x, f, GroupNormLinf(0.01, groups, [1, 0, ones(length(groups) - 2)...]), CDSolver(; tol = 1e-10))
  @test all(==(1), Vector(x)[6:10])              # an unpenalized group keeps its values
end

@testset "allocations" begin
  X, y = groupdata()
  n, p = size(X)
  groups = [i:i+4 for i in 1:5:p]
  # the group curvatures, which initialize! resets, are kept from the first run:
  # computing them allocates (ProximalSolvers #20), the steps must not
  function passes_bytes(x, m, g, s)
    PS.initialize!(m, x)
    st = PS._CDState(Float64, x, s.random)
    PS._cd!(x, m, g, st, s)
    L = copy(m.L)
    fill!(x, 0.0); PS.initialize!(m, x)
    copyto!(m.L, L)
    @allocated PS._cd!(x, m, g, st, s)
  end
  for f in ((1 / n) * LeastSquaresLoss(y, X), QuadraticFunction(X' * X / n, -X' * y / n)),
      g in (GroupNormLinf(0.05, groups), NormL1(0.05)), s in (CDSolver(; order = :cyclic), CDSolver(; order = :random))
    x = AtomIterate(Float64, p, groups)
    @test passes_bytes(x, cdmodel(f, x), g, s) == 0
  end
end

end
