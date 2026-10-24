module SparseXTest

using Test
using Random
using LinearAlgebra
using SparseArrays
using Statistics
using ProximalBase
using ProximalSolvers

const PS = ProximalSolvers

# a sparse design with a constant column stored in full and an empty column,
# which a model with an intercept must center exactly
function sparsedata(rng; n = 120, p = 200, density = 0.05, T = Float64)
  Xs = sprandn(rng, T, n, p, density)
  Xs[:, 3] .= T(2)
  Xs[:, 4] .= zero(T)
  dropzeros!(Xs)
  Xs[:, 3] .= T(2)
  β = zeros(T, p)
  β[[1, 2, 5, 6, 7]] .= randn(rng, T, 5)
  y = Matrix(Xs) * β + T(0.1) * randn(rng, T, n) .+ one(T)
  Xs, y
end

# solves with the dense and the sparse design, from the same seed
function both(f::Function)
  out = map(f, (:dense, :sparse))
  out[1], out[2]
end

@testset "least squares: sparse X takes the steps of dense X" begin
  rng = Xoshiro(1)
  Xs, y = sparsedata(rng)
  X = Matrix(Xs)
  n, p = size(X)
  @test nnz(Xs[:, 3]) == n && nnz(Xs[:, 4]) == 0
  λ = 0.1 * norm(X' * (y .- mean(y)), Inf) / n
  w = rand(Xoshiro(2), n) .+ 0.5
  for (intercept, weights) in ((false, false), (true, false), (false, true), (true, true)),
      solver in (CDSolver(; order = :cyclic), CDSolver(), CDSolver(; order = :cyclic, anderson = 5), CDSolver(; momentum = false))
    d, s = both() do kind
      A = kind === :dense ? X : Xs
      f = weights ? (1 / n) * GLMLoss(ProximalBase.Squared(), y, A; weights = w) : (1 / n) * LeastSquaresLoss(y, A)
      x = SparseIterate(p)
      m = cdmodel(f, x; intercept)
      Random.seed!(7)
      r = solve!(x, m, NormL1(λ), solver)
      G = zeros(p)
      PS.gradient!(G, m, x)
      (x = Vector(x), passes = r.passes, β0 = PS.intercept(m, x), res = copy(PS.residuals(m)),
       loss = PS.loss_value(m, x), G = G)
    end
    @test d.passes == s.passes
    @test d.x ≈ s.x atol = 1e-12
    @test d.β0 ≈ s.β0 atol = 1e-12
    @test d.res ≈ s.res atol = 1e-12
    @test d.loss ≈ s.loss rtol = 1e-12
    @test d.G ≈ s.G atol = 1e-12
    # the residuals are those of the fit
    @test s.res ≈ y .- s.β0 .- X * s.x atol = 1e-10
  end
end

@testset "square-root lasso and the logistic loss" begin
  rng = Xoshiro(3)
  Xs, y = sparsedata(rng)
  X = Matrix(Xs)
  n, p = size(X)
  wq = rand(Xoshiro(6), n) .+ 0.5
  for intercept in (false, true), weights in (nothing, wq)
    d, s = both() do kind
      f = SqrtLeastSquaresLoss(y, kind === :dense ? X : Xs; weights)
      x = SparseIterate(p)
      m = cdmodel(f, x; intercept)
      Random.seed!(3)
      r = solve!(x, m, NormL1(0.3), CDSolver())
      G = zeros(p)
      PS.gradient!(G, m, x)
      (x = Vector(x), passes = r.passes, res = copy(PS.residuals(m)), loss = PS.loss_value(m, x), G = G)
    end
    @test d.passes == s.passes
    @test d.x ≈ s.x atol = 1e-12
    @test d.res ≈ s.res atol = 1e-12
    @test d.loss ≈ s.loss rtol = 1e-12
    @test d.G ≈ s.G atol = 1e-10
  end
  yb = Float64.(y .> median(y))
  w, o = rand(Xoshiro(4), n) .+ 0.5, 0.1 * randn(Xoshiro(5), n)
  d, s = both() do kind
    A = kind === :dense ? X : Xs
    f = (1 / n) * LogisticLoss(yb, A; weights = w, offset = o)
    x1, x2 = SparseIterate(p), SparseIterate(p)
    Random.seed!(5)
    r = solve!(x1, f, NormL1(0.02), CDSolver())
    solve!(x2, f, NormL1(0.02), ProxNewton())
    (x1 = Vector(x1), passes = r.passes, x2 = Vector(x2))
  end
  @test d.passes == s.passes
  @test d.x1 ≈ s.x1 atol = 1e-12
  @test d.x2 ≈ s.x2 atol = 1e-10
  # proximal Newton with an intercept: its inner weighted least-squares model is
  # centered at weighted means, so the sparse X keeps its residuals as r .+ δ
  d, s = both() do kind
    A = kind === :dense ? X : Xs
    f = (1 / n) * LogisticLoss(yb, A; weights = w, offset = o)
    x = SparseIterate(p)
    m = PS.cdmodel(f, x; intercept = true)
    solve!(x, m, NormL1(0.02), ProxNewton(; tol = 1e-12, inner = CDSolver(; tol = 1e-12)))
    (x = Vector(x), β0 = PS.intercept(m, x))
  end
  # the two take different roundings, so they agree to the accuracy of the solves
  @test d.x ≈ s.x atol = 1e-8
  @test d.β0 ≈ s.β0 atol = 1e-8
end

@testset "paths, refits, groups and Float32" begin
  rng = Xoshiro(6)
  Xs, y = sparsedata(rng; p = 120)
  X = Matrix(Xs)
  n, p = size(X)
  f, fs = (1 / n) * LeastSquaresLoss(y, X), (1 / n) * LeastSquaresLoss(y, Xs)
  λmax = PS.lambda_max(f, NormL1(1.0), SparseIterate(p))
  λs = λmax .* 10 .^ range(0, -1.5; length = 15)
  Random.seed!(1)
  pd = solve_path(f, NormL1(1.0), λs)
  Random.seed!(1)
  ps = solve_path(fs, NormL1(1.0), λs)
  @test all(k -> isapprox(pd.solutions[k], ps.solutions[k]; atol = 1e-12), eachindex(λs))

  for intercept in (false, true)
    d, s = both() do kind
      g = kind === :dense ? f : fs
      x = SparseIterate(p)
      m = cdmodel(g, x; intercept)
      solve!(x, m, NormL1(0.2 * λmax), CDSolver(; order = :cyclic))
      refit!(x, m)
      Vector(x)
    end
    @test d ≈ s atol = 1e-10
  end

  groups = [collect(i:i + 3) for i in 1:4:p]
  gl = GroupNormL2(0.3 * λmax, groups)
  d, s = both() do kind
    x = AtomIterate(p, groups)
    r = solve!(x, kind === :dense ? f : fs, gl, CDSolver(; order = :cyclic))
    (Vector(x), r.passes)
  end
  @test d[2] == s[2]
  @test d[1] ≈ s[1] atol = 1e-12

  X32s, y32 = sparsedata(Xoshiro(7); T = Float32)
  n32 = size(X32s, 1)
  d, s = both() do kind
    A = kind === :dense ? Matrix(X32s) : X32s
    x = SparseIterate(Float32, size(A, 2))
    m = cdmodel((1f0 / n32) * LeastSquaresLoss(y32, A), x; intercept = true)
    Random.seed!(2)
    solve!(x, m, NormL1(0.05f0), CDSolver())
    (Vector(x), PS.residuals(m))
  end
  @test eltype(s[1]) == Float32
  @test d[1] ≈ s[1] atol = 1e-5
  @test d[2] ≈ s[2] atol = 1e-4
end

@testset "no allocations in the passes" begin
  rng = Xoshiro(8)
  Xs, y = sparsedata(rng)
  n, p = size(Xs)
  function passes_bytes(x, m, g, s)
    PS.initialize!(m, x)
    st = PS._CDState(Float64, x, s.random)
    PS._cd!(x, m, g, st, s)
    fill!(x, 0.0); PS.initialize!(m, x)
    @allocated PS._cd!(x, m, g, st, s)
  end
  f = (1 / n) * LeastSquaresLoss(y, Xs)
  yb = Float64.(y .> median(y))
  for s in (CDSolver(; order = :cyclic), CDSolver(), CDSolver(; momentum = true)), intercept in (false, true)
    @test passes_bytes(SparseIterate(p), cdmodel(f, SparseIterate(p); intercept), NormL1(0.05), s) == 0
    @test passes_bytes(SparseIterate(p), cdmodel(SqrtLeastSquaresLoss(y, Xs), SparseIterate(p); intercept), NormL1(0.3), s) == 0
  end
  @test passes_bytes(SparseIterate(p), cdmodel((1 / n) * LogisticLoss(yb, Xs), SparseIterate(p)), NormL1(0.02), CDSolver()) == 0
end

end
