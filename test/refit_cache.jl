# refit! with a RefitCache: the updated factorization gives the refits of a
# fresh one along random sequences of supports, and falls back where it must.

module RefitCacheTests

using Test, LinearAlgebra, Random, SparseArrays, Statistics
using ProximalBase
using ProximalSolvers
using ProximalSolvers: intercept

# supports that change by one to three coordinates at a time, in no particular order
function support_walk(rng, p, steps; start = 3, kmax = 40)
  S = randperm(rng, p)[1:start]
  walk = [copy(S)]
  for _ in 1:steps
    length(S) > 2 && rand(rng) < 0.4 && deleteat!(S, rand(rng, 1:length(S)))
    for _ in 1:rand(rng, 1:2)
      push!(S, rand(rng, setdiff(1:p, S)))
    end
    length(S) > kmax && deleteat!(S, 1)
    push!(walk, copy(S))
  end
  walk
end

# the refit from zero on the support S
function refit_on(f, p, S; cache = nothing, T = Float64)
  x = SparseIterate(T, p)
  refit!(x, f; support = S, cache)
  x
end

@testset "RefitCache" begin
  rng = Xoshiro(17)
  n, p = 200, 60
  X, y = randn(rng, n, p), randn(rng, n)
  walk = support_walk(rng, p, 60)

  @testset "least squares" begin
    f = (1 / n) * LeastSquaresLoss(y, X)
    cache = RefitCache()
    for S in walk
      @test Vector(refit_on(f, p, S; cache)) ≈ Vector(refit_on(f, p, S)) rtol = 1e-9
    end
    @test cache.updates > 2 * length(walk)
    @test cache.refactors <= 3
    # the refit is least squares on the support
    S = walk[end]
    @test Vector(refit_on(f, p, S; cache))[S] ≈ X[:, S] \ y
  end

  @testset "least squares with an intercept" begin
    y1 = y .+ 3
    x = SparseIterate(p)
    m = cdmodel((1 / n) * LeastSquaresLoss(y1, X), x; intercept = true)
    cache = RefitCache()
    for S in walk[1:20]
      xc, xf = SparseIterate(p), SparseIterate(p)
      refit!(xc, m; support = S, cache)
      refit!(xf, m; support = S)
      @test Vector(xc) ≈ Vector(xf) rtol = 1e-9
      coef = [ones(n) X[:, S]] \ y1
      @test Vector(xc)[S] ≈ coef[2:end] rtol = 1e-8
      @test intercept(m, Vector(xc)) ≈ coef[1] rtol = 1e-8
    end
  end

  @testset "weighted least squares with an intercept, and a sparse design" begin
    w = rand(rng, n) .+ 0.5
    Xs = sparse(X .* (rand(rng, n, p) .< 0.3))
    y1 = y .+ 3
    for (m, A, sw) in ((cdmodel(GLMLoss(ProximalBase.Squared(), y1, X; weights = w), SparseIterate(p); intercept = true),
                        X, sqrt.(w)),
                       (cdmodel((1 / n) * LeastSquaresLoss(y1, Xs), SparseIterate(p); intercept = true), Matrix(Xs), ones(n)))
      cache = RefitCache()
      for S in walk[1:20]
        xc, xf = SparseIterate(p), SparseIterate(p)
        refit!(xc, m; support = S, cache)
        refit!(xf, m; support = S)
        @test Vector(xc) ≈ Vector(xf) rtol = 1e-8
        coef = (sw .* [ones(n) A[:, S]]) \ (sw .* y1)
        @test Vector(xc)[S] ≈ coef[2:end] rtol = 1e-8
      end
      @test cache.updates > 0
    end
  end

  @testset "weighted least squares and quadratics" begin
    w = rand(rng, n) .+ 0.5
    fw = GLMLoss(ProximalBase.Squared(), y, X; weights = w)
    A = Matrix(Symmetric(X' * X / n)) + 0.1I
    fq = QuadraticFunction(A, -X' * y / n)
    for f in (fw, fq)
      cache = RefitCache()
      for S in walk[1:30]
        @test Vector(refit_on(f, p, S; cache)) ≈ Vector(refit_on(f, p, S)) rtol = 1e-9
      end
      @test cache.updates > 0
    end
  end

  @testset "a large change factors again" begin
    f = (1 / n) * LeastSquaresLoss(y, X)
    cache = RefitCache()
    refit_on(f, p, [1, 2, 3]; cache)
    r = cache.refactors
    S = collect(10:40)
    @test Vector(refit_on(f, p, S; cache)) ≈ Vector(refit_on(f, p, S)) rtol = 1e-9
    @test cache.refactors == r + 1
  end

  @testset "fallbacks" begin
    # a column that is a copy of another: the pivoted QR of refit! without the cache
    Xd = [X X[:, 5]]
    fd = (1 / n) * LeastSquaresLoss(y, Xd)
    cache = RefitCache()
    refit_on(fd, p + 1, [5, 9, 12]; cache)
    S = [5, 9, 12, p + 1]
    xc = refit_on(fd, p + 1, S; cache)
    @test Vector(xc) ≈ Vector(refit_on(fd, p + 1, S)) atol = 1e-10
    @test isempty(cache.S)                 # the cache starts over
    # another loss: the factor of the first is not used
    f1, f2 = (1 / n) * LeastSquaresLoss(y, X), (1 / n) * LeastSquaresLoss(y, 2X)
    cache = RefitCache()
    refit_on(f1, p, [1, 2, 3]; cache)
    @test Vector(refit_on(f2, p, [1, 2, 3, 4]; cache)) ≈ Vector(refit_on(f2, p, [1, 2, 3, 4])) rtol = 1e-9
    # a loss the cache does not serve, and a cache of another element type, are refit as without it
    q = 8
    Sx = cov(randn(rng, 100, q)); Sy = cov(randn(rng, 100, q))
    fk = KronQuadratic(Sx, Sy, Sx - Sy)
    xs, xk = SymmetricSparseIterate(q), SymmetricSparseIterate(q)
    refit!(xs, fk; support = [1, 3, 7], cache = RefitCache())
    refit!(xk, fk; support = [1, 3, 7])
    @test Matrix(xs) ≈ Matrix(xk)
    f32 = (1f0 / n) * LeastSquaresLoss(Float32.(y), Float32.(X))
    @test Vector(refit_on(f32, p, [2, 4]; cache = RefitCache(), T = Float32)) ≈
          Vector(refit_on(f32, p, [2, 4]; T = Float32))
  end
end

end
