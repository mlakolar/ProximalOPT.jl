# Coordinate descent: correctness against closed forms and KKT conditions
# computed from dense algebra, the behaviors carried over from
# CoordinateDescent.jl, and allocations.

module CDTests

using Test, LinearAlgebra, Random, SparseArrays, Statistics
using ProximalBase
using ProximalOPT
using ProximalOPT: intercept, residuals
const PO = ProximalOPT

# largest violation of 0 ∈ ∇ + λ ∂(Σ wᵢ|xᵢ|)
function kkt_l1(grad, x, λ, w = ones(length(x)))
  v = 0.0
  for i in eachindex(x)
    v = max(v, x[i] == 0 ? max(abs(grad[i]) - λ * w[i], 0.0) : abs(grad[i] + λ * w[i] * sign(x[i])))
  end
  v
end

lsgrad(X, y, x) = X' * (X * x - y) / size(X, 1)

function lassodata(; n = 80, p = 200, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  β = zeros(p); β[1:5] .= [2, -1.5, 1, -1, 0.5]
  X, X * β + 0.3 * randn(rng, n)
end

@testset "closed forms" begin
  # orthogonal columns: the lasso solution soft-thresholds X'y/n
  n = 64
  Q = Matrix(qr(randn(Xoshiro(2), n, 8)).Q)[:, 1:8] .* sqrt(n)   # ‖Xⱼ‖² = n, X'X = nI
  y = randn(Xoshiro(3), n)
  λ = 0.3
  z = Q' * y / n
  for order in (:cyclic, :random)
    x = SparseIterate(8)
    r = solve!(x, (1 / n) * LeastSquaresLoss(y, Q), NormL1(λ), CDSolver(; order, tol = 1e-12))
    @test Vector(x) ≈ sign.(z) .* max.(abs.(z) .- λ, 0) atol = 1e-12
    @test r.converged
  end
  # a group penalty on orthogonal columns shrinks each group's norm
  groups = [1:3, 4:5, 6:8]
  x = AtomIterate(Float64, 8, groups)
  solve!(x, (1 / n) * LeastSquaresLoss(y, Q), GroupNormL2(λ, groups, ones(3)), CDSolver(; tol = 1e-12))
  for G in groups
    @test Vector(x)[G] ≈ max(1 - λ / norm(z[G]), 0) * z[G] atol = 1e-12
  end
end

@testset "lasso and its variants" begin
  X, y = lassodata()
  n, p = size(X)
  λ = 0.1 * maximum(abs, X' * y) / n
  f = (1 / n) * LeastSquaresLoss(y, X)

  @testset "$order order" for order in (:cyclic, :random)
    x = SparseIterate(p)
    r = solve!(x, f, NormL1(λ), CDSolver(; order, tol = 1e-10))
    @test r isa CDResults && r.converged && r.minimizer === x
    @test kkt_l1(lsgrad(X, y, Vector(x)), Vector(x), λ) < 1e-7
    @test r.minimum ≈ sum(abs2, y - X * Vector(x)) / (2n) + λ * sum(abs, Vector(x))
  end

  @testset "unscaled loss and penalty" begin
    x1, x2 = SparseIterate(p), SparseIterate(p)
    solve!(x1, f, NormL1(λ), CDSolver(; tol = 1e-12))
    solve!(x2, LeastSquaresLoss(y, X), NormL1(n * λ), CDSolver(; tol = 1e-12))
    @test Vector(x1) ≈ Vector(x2) atol = 1e-9
  end

  @testset "a seeded random order is reproducible" begin
    run() = (Random.seed!(7); x = SparseIterate(p); r = solve!(x, f, NormL1(λ), CDSolver()); (Vector(x), r.passes))
    @test run() == run()
  end

  @testset "unpenalized coordinates" begin
    w = ones(p); w[[3, 50]] .= 0
    x = SparseIterate(p)
    solve!(x, f, NormL1(10 * λ, w), CDSolver(; tol = 1e-10))
    @test x[3] != 0 && x[50] != 0
    @test kkt_l1(lsgrad(X, y, Vector(x)), Vector(x), 10λ, w) < 1e-7
    # λmax ignores the unpenalized coordinates
    @test lambda_max(f, NormL1(1.0, w), SparseIterate(p)) ≈ maximum(abs.(X' * y / n)[w .> 0])
  end

  @testset "warm start from a dense starting point" begin
    x = SparseIterate(randn(Xoshiro(4), p))
    solve!(x, f, NormL1(λ), CDSolver(; tol = 1e-10))
    @test kkt_l1(lsgrad(X, y, Vector(x)), Vector(x), λ) < 1e-7
  end

  @testset "data changed in place" begin
    Xc, yc = copy(X), copy(y)
    fc = (1 / n) * LeastSquaresLoss(yc, Xc)
    m = cdmodel(fc, SparseIterate(p))
    x = SparseIterate(p)
    solve!(x, m, NormL1(λ), CDSolver(; tol = 1e-10))
    Xc .*= 2
    x = SparseIterate(p)
    solve!(x, m, NormL1(λ), CDSolver(; tol = 1e-10))   # initialize! rereads the columns
    @test kkt_l1(lsgrad(Xc, yc, Vector(x)), Vector(x), λ) < 1e-7
  end

  @testset "zero and constant columns" begin
    Xz = copy(X); Xz[:, 2] .= 0; Xz[:, 4] .= 3.0
    x = SparseIterate(p)
    r = solve!(x, (1 / n) * LeastSquaresLoss(y, Xz), NormL1(λ), CDSolver())
    @test x[2] == 0 && r.converged
    # with an intercept a constant column does not enter the loss
    x = SparseIterate(p)
    m = cdmodel((1 / n) * LeastSquaresLoss(y, Xz), x; intercept = true)
    solve!(x, m, NormL1(λ), CDSolver())
    @test x[2] == 0 && x[4] == 0
    # an unpenalized column the loss does not depend on goes to the penalty's
    # minimizer nearest zero, here the projection of 0 onto [1, 2]
    x = SparseIterate(p)
    lo = zeros(p); lo[2] = 1
    hi = fill(Inf, p); hi[2] = 2
    solve!(x, (1 / n) * LeastSquaresLoss(y, Xz), IndBox(lo, hi), CDSolver(; maxpasses = 50))
    @test x[2] == 1
  end

  @testset "the stopping rule does not depend on the units" begin
    run(c) = (Random.seed!(1); x = SparseIterate(p);
              r = solve!(x, (1 / n) * LeastSquaresLoss(c * y, c * X), NormL1(c^2 * λ), CDSolver()); (Vector(x), r.passes))
    # a power of 2 scales every number exactly
    (x1, k1), (x2, k2) = run(1.0), run(1024.0)
    @test k1 == k2
    @test x1 == x2
  end

  @testset "intercept" begin
    y1 = y .+ 3.0
    x = SparseIterate(p)
    m = cdmodel((1 / n) * LeastSquaresLoss(y1, X), x; intercept = true)
    solve!(x, m, NormL1(λ), CDSolver(; tol = 1e-10))
    b0 = intercept(m, Vector(x))
    res = y1 .- b0 .- X * Vector(x)
    @test abs(sum(res)) < 1e-8
    @test residuals(m) ≈ res
    @test kkt_l1(-X' * res / n, Vector(x), λ) < 1e-7
  end

  @testset "weights and offset" begin
    w = rand(Xoshiro(5), n) .+ 0.5
    o = randn(Xoshiro(6), n)
    ω = rand(Xoshiro(7), p) .+ 0.5
    x = SparseIterate(p)
    solve!(x, GLMLoss(ProximalBase.Squared(), y, X; weights = w, offset = o), NormL1(n * λ, ω), CDSolver(; tol = 1e-10))
    @test kkt_l1(X' * (w .* (X * Vector(x) + o - y)), Vector(x), n * λ, ω) < 1e-6
  end

  @testset "quadratic" begin
    A = Matrix(Symmetric(X' * X / n)) + 0.1I
    b = -X' * y / n
    x = SparseIterate(p)
    r = solve!(x, QuadraticFunction(A, b), NormL1(λ), CDSolver(; tol = 1e-10))
    @test kkt_l1(A * Vector(x) + b, Vector(x), λ) < 1e-8
    @test r.minimum ≈ dot(Vector(x), A * Vector(x)) / 2 + dot(b, Vector(x)) + λ * sum(abs, Vector(x))
    # a Symmetric matrix and a scaled quadratic
    x2 = SparseIterate(p)
    solve!(x2, 2 * QuadraticFunction(Symmetric(A), b), NormL1(2λ), CDSolver(; tol = 1e-10))
    @test Vector(x2) ≈ Vector(x) atol = 1e-8
  end

  @testset "other separable penalties" begin
    for g in (ElasticNet(λ, 0.5λ), MCP(λ, 3.0), SCAD(λ, 3.7), NormL1Nonneg(λ), IndBox(-0.5, 0.5), ZeroFunction())
      x = SparseIterate(p)
      r = solve!(x, f, g, CDSolver(; tol = 1e-10, maxpasses = 10^5))
      @test r.converged
      @test isfinite(r.minimum)
    end
    # nonnegative lasso: KKT with the sign constraint
    x = SparseIterate(p)
    solve!(x, f, NormL1Nonneg(λ), CDSolver(; tol = 1e-10))
    xd = Vector(x); gr = lsgrad(X, y, xd)
    @test all(xd .>= 0)
    @test maximum(i -> xd[i] > 0 ? abs(gr[i] + λ) : max(-gr[i] - λ, 0.0), 1:p) < 1e-7
  end

  @testset "Float32" begin
    X32, y32 = Float32.(X), Float32.(y)
    x = SparseIterate(Float32, p)
    r = solve!(x, (1f0 / n) * LeastSquaresLoss(y32, X32), NormL1(Float32(λ)), CDSolver(; tol = 1e-5))
    @test eltype(x) == Float32 && r.minimum isa Float32
    x64 = SparseIterate(p)
    solve!(x64, f, NormL1(λ), CDSolver(; tol = 1e-10))
    @test Vector(x) ≈ Vector(x64) atol = 1e-3
  end

  @testset "Anderson acceleration" begin
    x0, x5 = SparseIterate(p), SparseIterate(p)
    r0 = solve!(x0, f, NormL1(0.2λ), CDSolver(; order = :cyclic, tol = 1e-10))
    r5 = solve!(x5, f, NormL1(0.2λ), CDSolver(; order = :cyclic, tol = 1e-10, anderson = 5))
    @test r5.converged && r5.passes < r0.passes
    @test kkt_l1(lsgrad(X, y, Vector(x5)), Vector(x5), 0.2λ) < 1e-7
    @test Vector(x5) ≈ Vector(x0) atol = 1e-6
    # ignored with a random order
    run(K) = (Random.seed!(3); x = SparseIterate(p); solve!(x, f, NormL1(λ), CDSolver(; anderson = K)).passes)
    @test run(5) == run(0)
    @test_throws ArgumentError CDSolver(; anderson = -1)
  end

  @testset "maxpasses" begin
    x = SparseIterate(p)
    r = @test_logs (:warn, r"did not converge") solve!(x, f, NormL1(0.01λ), CDSolver(; maxpasses = 3))
    @test !r.converged && r.passes == 3
  end
end

@testset "group lasso" begin
  X, y = lassodata()
  n, p = size(X)
  groups = [i:i+4 for i in 1:5:p]
  λ = 0.1 * maximum(G -> norm(X[:, G]' * y), groups) / n
  g = GroupNormL2(λ, groups, ones(length(groups)))
  for f in ((1 / n) * LeastSquaresLoss(y, X), QuadraticFunction(X' * X / n, -X' * y / n))
    x = AtomIterate(Float64, p, groups)
    solve!(x, f, g, CDSolver(; tol = 1e-10))
    xd = Vector(x); gr = lsgrad(X, y, xd)
    @test maximum(G -> norm(xd[G]) == 0 ? max(norm(gr[G]) - λ, 0) : norm(gr[G] + λ * xd[G] / norm(xd[G])), groups) < 1e-6
  end
  @test_throws ArgumentError solve!(SparseIterate(p), (1 / n) * LeastSquaresLoss(y, X), g, CDSolver())
  @test_throws ArgumentError solve!(AtomIterate(Float64, p, [1:p]), (1 / n) * LeastSquaresLoss(y, X), g, CDSolver())
end

@testset "KronQuadratic" begin
  rng = Xoshiro(11)
  q = 8
  Sx = cov(randn(rng, 40, q)); Sy = cov(randn(rng, 40, q))
  C = randn(rng, q, q)
  f = KronQuadratic(Sx, Sy, C)
  K = kron(Sy, Sx)

  @testset "value and gradient" begin
    Θ = randn(rng, q, q)
    @test f(Θ) ≈ dot(vec(Θ), K * vec(Θ)) / 2 + dot(C, Θ)
    @test f(vec(Θ)) ≈ f(Θ)
    G = similar(Θ)
    @test ProximalBase.value_and_gradient!(G, f, Θ) ≈ f(Θ)
    @test vec(G) ≈ K * vec(Θ) + vec(C)
    S = SymmetricSparseIterate(Matrix(Symmetric(Θ)))
    gradient!(G, f, S)
    D = Matrix(S)
    @test G ≈ (Sx * D * Sy + Sy * D * Sx) / 2 + (C + C') / 2
    @test f(S) ≈ f(D)
    @test_throws DimensionMismatch f(randn(q, q + 1))
    @test_throws ArgumentError KronQuadratic(randn(3, 3), Sy)
  end

  @testset "coordinate descent on vec Θ" begin
    ω = fill(0.05, q * q)
    x = SparseIterate(q * q)
    r = solve!(x, f, NormL1(1.0, ω), CDSolver(; tol = 1e-12))
    θ = Vector(x)
    @test kkt_l1(K * θ + vec(C), θ, 1.0, ω) < 1e-9
    @test r.minimum ≈ f(θ) + sum(ω .* abs.(θ))
    # the same as a QuadraticFunction of vec Θ with the Kronecker product formed
    x2 = SparseIterate(q * q)
    solve!(x2, QuadraticFunction(K, vec(C)), NormL1(1.0, ω), CDSolver(; tol = 1e-12))
    @test Vector(x2) ≈ θ atol = 1e-8
    # a matrix iterate is the same
    x3 = SparseIterate(q, q)
    solve!(x3, f, NormL1(1.0, reshape(ω, q, q)), CDSolver(; tol = 1e-12))
    @test vec(Matrix(x3)) ≈ θ atol = 1e-8
  end

  @testset "coordinate descent on a symmetric Θ" begin
    Cs = (C + C') / 2
    fs = KronQuadratic(Sx, Sy, Cs)
    λ = 0.3 * maximum(abs, Cs)
    for order in (:cyclic, :random)
      x = SymmetricSparseIterate(q)
      r = solve!(x, fs, NormL1(λ), CDSolver(; order, tol = 1e-12))
      D = Matrix(x)
      gr = (Sx * D * Sy + Sy * D * Sx) / 2 + Cs
      @test kkt_l1(vec(gr), vec(D), λ) < 1e-9
      @test r.minimum ≈ fs(D) + λ * sum(abs, D)
    end
    # Anderson acceleration on a symmetric iterate
    x = SymmetricSparseIterate(q)
    solve!(x, fs, NormL1(λ), CDSolver(; order = :cyclic, anderson = 5, tol = 1e-12))
    D = Matrix(x)
    @test kkt_l1(vec((Sx * D * Sy + Sy * D * Sx) / 2 + Cs), vec(D), λ) < 1e-9
    # weights on the entries, and an unpenalized diagonal
    W = fill(1.0, q, q); W[diagind(W)] .= 0
    x = SymmetricSparseIterate(q)
    solve!(x, fs, NormL1(λ, W), CDSolver(; tol = 1e-12))
    D = Matrix(x)
    @test all(!iszero, diag(D))
    @test kkt_l1(vec((Sx * D * Sy + Sy * D * Sx) / 2 + Cs), vec(D), λ, vec(W)) < 1e-9
  end
end

@testset "paths, continuation and λmax" begin
  X, y = lassodata()
  n, p = size(X)
  f = (1 / n) * LeastSquaresLoss(y, X)
  λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))
  @test λmax ≈ maximum(abs, X' * y) / n
  x = SparseIterate(p)
  solve!(x, f, NormL1(λmax * (1 + 1e-9)), CDSolver())
  @test nnz(x) == 0

  λ = 0.05λmax
  x = SparseIterate(p)
  r = solve!(x, f, NormL1(λ), CDSolver(; tol = 1e-10); continuation = 20)
  @test kkt_l1(lsgrad(X, y, Vector(x)), Vector(x), λ) < 1e-7
  @test r.kkt_checks >= 21

  λs = λmax .* 10 .^ range(0, -2; length = 15)
  for screening in (true, false)
    path = solve_path(f, NormL1(1.0), λs, CDSolver(; tol = 1e-10); screening)
    @test path.λ == λs
    for k in eachindex(λs)
      b = Vector(path.solutions[k])
      @test kkt_l1(lsgrad(X, y, b), b, λs[k]) < 1e-6
      @test path.values[k] ≈ sum(abs2, y - X * b) / (2n) + λs[k] * sum(abs, b)
    end
    @test all(path.converged)
    @test screening ? all(>=(1), path.kkt_checks) : all(iszero, path.kkt_checks)
  end
  # early stops
  @test length(solve_path(f, NormL1(1.0), λs; maxnnz = 10).λ) < length(λs)
  @test length(solve_path(f, NormL1(1.0), λs; callback = (k, λ, x, m) -> k == 3).λ) == 3
  @test_throws ArgumentError solve_path(f, NormL1(1.0), reverse(λs))
  @test_throws ArgumentError solve_path(f, NormL1(1.0), [-1.0])
  # paths of other penalties with a level
  for g in (ElasticNet(1.0, 0.1), MCP(1.0, 3.0), SCAD(1.0, 3.7))
    @test length(solve_path(f, g, λs[1:5]).λ) == 5
  end
  @test_throws ArgumentError solve_path(f, IndBox(-1, 1), λs)

  @testset "the strong rule is corrected by the optimality checks" begin
    # a coordinate whose gradient is small at λmax but large at the next level
    # is discarded by the strong rule and added back by the check
    rng = Xoshiro(21)
    Xs = randn(rng, 100, 30)
    ys = Xs[:, 1] - 0.9 * Xs[:, 2] + 0.01 * randn(rng, 100)
    fs = (1 / 100) * LeastSquaresLoss(ys, Xs)
    lm = lambda_max(fs, NormL1(1.0), SparseIterate(30))
    path = solve_path(fs, NormL1(1.0), [lm, 0.05lm], CDSolver(; tol = 1e-10))
    b = Vector(path.solutions[2])
    @test kkt_l1(lsgrad(Xs, ys, b), b, 0.05lm) < 1e-7
  end

  @testset "symmetric iterates" begin
    rng = Xoshiro(12)
    q = 12
    Sx = cov(randn(rng, 80, q)); Sy = cov(randn(rng, 80, q) * cholesky([0.6^abs(i - j) for i in 1:q, j in 1:q]).U)
    fk = KronQuadratic(Sx, Sy, Sx - Sy)
    lm = lambda_max(fk, NormL1(1.0), SymmetricSparseIterate(q))
    @test lm ≈ maximum(abs, Sy - Sx)
    for screening in (true, false)
      pk = solve_path(fk, NormL1(1.0), lm .* 10 .^ range(0, -1; length = 6), CDSolver(; tol = 1e-12);
                      x = SymmetricSparseIterate(q), screening)
      for k in eachindex(pk.λ)
        D = Matrix(pk.solutions[k])
        @test kkt_l1(vec((Sx * D * Sy + Sy * D * Sx) / 2 + (Sx - Sy)), vec(D), pk.λ[k]) < 1e-8
      end
    end
  end
end

@testset "square-root lasso" begin
  X, y = lassodata()
  n, p = size(X)
  λ = 0.5 * maximum(abs, X' * y) / norm(y)
  for intercept in (false, true)
    x = SparseIterate(p)
    m = cdmodel(SqrtLeastSquaresLoss(y, X), x; intercept)
    r = solve!(x, m, NormL1(λ), CDSolver(; tol = 1e-12))
    b0 = intercept ? PO.intercept(m, Vector(x)) : 0.0
    res = y .- b0 .- X * Vector(x)
    @test kkt_l1(-X' * res / norm(res), Vector(x), λ) < 1e-7
    @test r.minimum ≈ norm(res) + λ * sum(abs, Vector(x))
  end
  # near a perfect fit the step stays finite
  yp = X[:, 1] * 2.0
  x = SparseIterate(p)
  r = solve!(x, SqrtLeastSquaresLoss(yp, X), NormL1(1e-3), CDSolver(; maxpasses = 1000))
  @test all(isfinite, Vector(x))
  @test_throws ArgumentError solve!(SparseIterate(p), SqrtLeastSquaresLoss(y, X), MCP(1.0, 3.0), CDSolver())
end

@testset "reweighting" begin
  X, y = lassodata()
  n, p = size(X)
  f = (1 / n) * LeastSquaresLoss(y, X)
  λ0 = sqrt(2 * log(p) / n)
  # the scaled lasso: the level λ0 σ̂ at the fixed point σ̂ = ‖r‖/√n
  update = (g, x, m) -> (σ = norm(residuals(m)) / sqrt(n); (NormL1(λ0 * σ), abs(σ - g.λ0 / λ0) / (g.λ0 / λ0)))
  x = SparseIterate(p)
  res = reweight!(x, f, NormL1(λ0 * std(y)), update, CDSolver(; tol = 1e-12); tol = 1e-10, maxiter = 100)
  @test res.converged && res.cd_converged
  σ = norm(y - X * Vector(x)) / sqrt(n)
  @test kkt_l1(lsgrad(X, y, Vector(x)), Vector(x), λ0 * σ) < 1e-7
  @test res.penalty.λ0 ≈ λ0 * σ rtol = 1e-8
  # two stages with weights from the first solution
  stages = Ref(0)
  update2 = (g, x, m) -> (stages[] += 1; (NormL1(g.λ0, 1 .+ abs.(Vector(x))), Inf))
  res2 = reweight!(SparseIterate(p), f, NormL1(0.1), update2; maxiter = 2, tol = 0)
  @test res2.iterations == 2 && stages[] == 2 && !res2.converged
end

@testset "refit" begin
  X, y = lassodata()
  n, p = size(X)
  λ = 0.1 * maximum(abs, X' * y) / n
  f = (1 / n) * LeastSquaresLoss(y, X)
  x = SparseIterate(p)
  solve!(x, f, NormL1(λ), CDSolver())
  S = findall(!iszero, Vector(x))
  refit!(x, f)
  @test Vector(x)[S] ≈ X[:, S] \ y
  @test all(iszero, Vector(x)[setdiff(1:p, S)])
  # with an intercept
  y1 = y .+ 2
  x = SparseIterate(p)
  m = cdmodel((1 / n) * LeastSquaresLoss(y1, X), x; intercept = true)
  solve!(x, m, NormL1(λ), CDSolver())
  S = findall(!iszero, Vector(x))
  refit!(x, m)
  coef = [ones(n) X[:, S]] \ y1
  @test Vector(x)[S] ≈ coef[2:end]
  @test intercept(m, Vector(x)) ≈ coef[1]
  # quadratic, and a given support
  A = Matrix(Symmetric(X' * X / n)) + 0.1I; b = -X' * y / n
  xq = SparseIterate(p)
  refit!(xq, QuadraticFunction(A, b); support = [1, 2, 7])
  @test Vector(xq)[[1, 2, 7]] ≈ -(A[[1, 2, 7], [1, 2, 7]] \ b[[1, 2, 7]])
  # symmetric Kronecker: the gradient vanishes on the support
  rng = Xoshiro(13)
  q = 10
  Sx = cov(randn(rng, 100, q)); Sy = cov(randn(rng, 100, q))
  fk = KronQuadratic(Sx, Sy, Sx - Sy)
  xs = SymmetricSparseIterate(q)
  solve!(xs, fk, NormL1(0.3 * maximum(abs, Sy - Sx)), CDSolver())
  refit!(xs, fk)
  D = Matrix(xs)
  @test maximum(abs, ((Sx * D * Sy + Sy * D * Sx) / 2 + (Sx - Sy))[D .!= 0]) < 1e-10
end

# the quadratic x'Ax/2 + b'x through the generic model interface only
mutable struct MyQuadratic <: CDModel{Float64}
  A::Matrix{Float64}
  b::Vector{Float64}
  u::Vector{Float64}
end
PO.initialize!(m::MyQuadratic, x) = (m.u .= m.A * Vector(x); m)
PO._coordinate(m::MyQuadratic, x, k::Int) = (m.u[k] + m.b[k], m.A[k, k])
PO._update!(m::MyQuadratic, x, k::Int, h) = (m.u .+= h .* view(m.A, :, k); m)
PO._coordscale(m::MyQuadratic, x, k::Int) = sqrt(m.A[k, k])
PO._responsescale(m::MyQuadratic, x) = 1.0
PO._value(m::MyQuadratic, x) = dot(Vector(x), m.u) / 2 + dot(m.b, Vector(x))
ProximalBase.gradient!(G, m::MyQuadratic, x) = (G .= m.u .+ m.b)

@testset "user-defined model" begin
  rng = Xoshiro(14)
  B = randn(rng, 30, 20); A = B' * B / 30; b = randn(rng, 20)
  x = SparseIterate(20)
  r = solve!(x, MyQuadratic(A, b, zeros(20)), NormL1(0.2), CDSolver(; tol = 1e-12))
  @test kkt_l1(A * Vector(x) + b, Vector(x), 0.2) < 1e-9
  # and a path through it
  path = solve_path(MyQuadratic(A, b, zeros(20)), NormL1(1.0), [1.0, 0.5, 0.2]; x = SparseIterate(20))
  @test Vector(path.solutions[3]) ≈ Vector(x) atol = 1e-6
end

@testset "argument checks" begin
  X, y = lassodata()
  n, p = size(X)
  f = (1 / n) * LeastSquaresLoss(y, X)
  @test_throws ArgumentError CDSolver(; tol = -1)
  @test_throws ArgumentError CDSolver(; maxpasses = -1)
  @test_throws ArgumentError CDSolver(; order = :sideways)
  @test_throws ArgumentError solve!(SparseIterate(p), f, NormL2(1.0), CDSolver())        # not separable
  @test_throws DimensionMismatch solve!(SparseIterate(p + 1), f, NormL1(1.0), CDSolver())
  @test_throws ArgumentError solve!(SparseIterate(p), NormL2(1.0), NormL1(1.0), CDSolver()) # no model
  Xn = copy(X); Xn[1, 1] = NaN
  @test_throws ArgumentError solve!(SparseIterate(p), LeastSquaresLoss(y, Xn), NormL1(1.0), CDSolver())
  @test_throws ArgumentError solve!(SparseIterate(p), f, NormL1(1.0), CDSolver(); continuation = -1)
end

@testset "allocations" begin
  X, y = lassodata()
  n, p = size(X)
  f = (1 / n) * LeastSquaresLoss(y, X)
  # the passes allocate nothing once the model and the state exist
  function passes_bytes(x, m, g, s)
    PO.initialize!(m, x)
    st = PO._CDState(Float64, x, s.random)
    PO._cd!(x, m, g, st, s)
    fill!(x, 0.0); PO.initialize!(m, x)
    @allocated PO._cd!(x, m, g, st, s)
  end
  for s in (CDSolver(; order = :cyclic), CDSolver(; order = :random))
    @test passes_bytes(SparseIterate(p), cdmodel(f, SparseIterate(p)), NormL1(0.1), s) == 0
    q = 10
    Sx = cov(randn(Xoshiro(15), 50, q)); Sy = cov(randn(Xoshiro(16), 50, q))
    fk = KronQuadratic(Sx, Sy, Sx - Sy)
    @test passes_bytes(SymmetricSparseIterate(q), cdmodel(fk, SymmetricSparseIterate(q)), NormL1(0.05), s) == 0
    @test passes_bytes(SparseIterate(q * q), cdmodel(fk, SparseIterate(q * q)), NormL1(0.05), s) == 0
  end
end

end
