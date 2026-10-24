# Duality gaps: weak duality at arbitrary points and a zero gap at accurate
# solutions, gap stopping in every solver, gap-safe screening, the KKT residual,
# and allocations.

module GapTests

using Test, LinearAlgebra, Random, Statistics
using ProximalBase
using ProximalSolvers
const PS = ProximalSolvers

function lassodata(; n = 80, p = 200, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  β = zeros(p); β[1:5] .= [2, -1.5, 1, -1, 0.5]
  X, X * β + 0.3 * randn(rng, n), β
end

# a sparse point with random signs, for weak duality
randpoint(rng, p; nonneg = false) = SparseIterate((nonneg ? rand(rng, p) : randn(rng, p)) .* (rand(rng, p) .< 0.1))

relgap(dg) = dg.gap / abs(dg.primal)

@testset "weak duality, and a zero gap at the solution" begin
  X, y, β = lassodata()
  n, p = size(X)
  rng = Xoshiro(7)
  w = rand(rng, n) .+ 0.5
  o = 0.1 * randn(rng, n)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  wls = GLMLoss(ProximalBase.Squared(), y, X; weights = w ./ n, offset = o)
  λmax = lambda_max(ls, NormL1(1.0), SparseIterate(p))
  yb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X * β)))
  logit = (1 / n) * GLMLoss(ProximalBase.Logistic(), yb, X)
  huber = (1 / n) * GLMLoss(ProximalBase.Huber(0.5), y, X)
  λl = lambda_max(logit, NormL1(1.0), SparseIterate(p))
  cases = Any[(ls, NormL1(0.1λmax)), (ls, NormL1(0.1λmax, rand(rng, p) .+ 0.5)), (ls, NormL1Nonneg(0.1λmax)),
              (ls, ElasticNet(0.1λmax, 0.05)), (ls, ElasticNet(0.1λmax, 0.0)), (wls, NormL1(0.05λmax)),
              (logit, NormL1(0.2λl)), (logit, ElasticNet(0.2λl, 0.01)), (huber, NormL1(0.1λmax))]
  for (f, g) in cases
    x = SparseIterate(p)
    solve!(x, f, g, CDSolver(; tol = 1e-14))
    dg = duality_gap(f, g, x)
    @test dg.gap ≈ dg.primal - dg.dual
    @test dg.primal ≈ f(x) + g(x)
    @test abs(relgap(dg)) < 1e-10
    Pstar = dg.primal
    for _ in 1:5
      xr = randpoint(rng, p; nonneg = g isa NormL1Nonneg)
      d = duality_gap(f, g, xr)
      @test d.gap >= d.primal - Pstar - 1e-10 * abs(Pstar)
      @test d.dual <= Pstar + 1e-10 * abs(Pstar)
      # a dense vector gives the same gap
      @test duality_gap(f, g, Vector(xr)).gap ≈ d.gap
    end
  end
  # with negative entries the nonnegative lasso is infinite at x
  @test duality_gap(ls, NormL1Nonneg(0.1λmax), SparseIterate([-1.0; zeros(p - 1)])).gap == Inf

  # an intercept, through the model
  m = cdmodel(ls, SparseIterate(p); intercept = true)
  x = SparseIterate(p)
  solve!(x, m, NormL1(0.1λmax), CDSolver(; tol = 1e-14))
  dg = duality_gap(m, NormL1(0.1λmax), x)
  @test abs(relgap(dg)) < 1e-10
  @test duality_gap(m, NormL1(0.1λmax), SparseIterate(p)).gap > 0

  # Poisson, by proximal Newton
  ycount = Float64.(rand(rng, 0:3, n))
  fp = (1 / n) * GLMLoss(ProximalBase.Poisson(), ycount, X ./ 4)
  x = SparseIterate(p)
  solve!(x, fp, NormL1(0.05), ProxNewton(; tol = 1e-14, inner = CDSolver(; tol = 1e-12)))
  @test abs(relgap(duality_gap(fp, NormL1(0.05), x))) < 1e-10
  @test duality_gap(fp, NormL1(0.05), randpoint(rng, p)).gap > 0

  # the group lasso on an AtomIterate
  groups = [((k - 1) * 4 + 1):(k * 4) for k in 1:50]
  gg = GroupNormL2(0.1λmax, groups)
  xa = AtomIterate(p, groups)
  solve!(xa, ls, gg, CDSolver(; tol = 1e-14))
  dg = duality_gap(ls, gg, xa)
  @test abs(relgap(dg)) < 1e-10
  @test duality_gap(ls, gg, Vector(xa)).gap ≈ dg.gap atol = 1e-12
  xr = AtomIterate(Vector(randpoint(rng, p)), groups)
  @test duality_gap(ls, gg, xr).gap >= duality_gap(ls, gg, xr).primal - dg.primal - 1e-10
end

@testset "no gap" begin
  X, y, _ = lassodata()
  n, p = size(X)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  q = QuadraticFunction(X'X / n, -X'y / n)
  @test_throws ArgumentError duality_gap(q, NormL1(0.1), SparseIterate(p))
  @test_throws ArgumentError duality_gap(ls, MCP(0.1), SparseIterate(p))
  @test_throws ArgumentError duality_gap(cdmodel(q, SparseIterate(p)), NormL1(0.1), SparseIterate(p))
  # a coordinate without penalty and a nonzero gradient
  w = ones(p); w[1] = 0
  @test_throws ArgumentError duality_gap(ls, NormL1(0.1, w), SparseIterate(p))
  # gaptol without a gap: the rule on the changes applies
  r = @test_logs (:warn, r"no duality gap") match_mode = :any solve!(SparseIterate(p), q, NormL1(0.1), CDSolver(; gaptol = 1e-8))
  @test r.converged && isnan(r.gap)
  @test isnan(solve!(SparseIterate(p), ls, NormL1(0.1), CDSolver()).gap)
end

@testset "gaptol in coordinate descent" begin
  X, y, β = lassodata()
  n, p = size(X)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  λmax = lambda_max(ls, NormL1(1.0), SparseIterate(p))
  rng = Xoshiro(3)
  yb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X * β)))
  logit = (1 / n) * GLMLoss(ProximalBase.Logistic(), yb, X)
  λl = lambda_max(logit, NormL1(1.0), SparseIterate(p))
  for gaptol in (1e-4, 1e-8, 1e-11), (f, g) in ((ls, NormL1(0.05λmax)), (logit, NormL1(0.1λl)))
    for s in (CDSolver(; gaptol), CDSolver(; gaptol, momentum = false), CDSolver(; gaptol, order = :cyclic),
              CDSolver(; gaptol, order = :cyclic, anderson = 5), CDSolver(; gaptol, gapsafe = true))
      x = SparseIterate(p)
      r = solve!(x, f, g, s)
      @test r.converged
      @test r.gap <= gaptol
      @test relgap(duality_gap(f, g, x)) <= gaptol * (1 + 1e-6) + 1e-14
    end
  end
  # a tolerance at or below the precision ends the run once tol cannot be
  # tightened further, converged only if rounding made the gap zero
  r = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
    solve!(SparseIterate(p), ls, NormL1(0.05λmax), CDSolver(; gaptol = 1e-30))
  end
  @test r.passes < 10_000
  @test r.converged == (r.gap <= 1e-30)
  # groups
  groups = [((k - 1) * 4 + 1):(k * 4) for k in 1:50]
  gg = GroupNormL2(0.1λmax, groups)
  for gapsafe in (false, true)
    xa = AtomIterate(p, groups)
    r = solve!(xa, ls, gg, CDSolver(; gaptol = 1e-9, gapsafe))
    @test r.gap <= 1e-9
    @test relgap(duality_gap(ls, gg, xa)) <= 1e-9 * (1 + 1e-6)
  end
  # through continuation and paths, which run the same passes
  x = SparseIterate(p)
  r = solve!(x, ls, NormL1(0.05λmax), CDSolver(; gaptol = 1e-9); continuation = 10)
  @test r.converged && r.gap <= 1e-9
  @test relgap(duality_gap(ls, NormL1(0.05λmax), x)) <= 1e-9 * (1 + 1e-6)
  # the levels before the target take continuation_tol and no gap rule; every
  # other field of the solver is kept
  s = CDSolver(; tol = 1e-9, order = :cyclic, anderson = 3, gaptol = 1e-9, gapsafe = true, maxpasses = 77)
  s2 = PS._with_tol(s, 1e-4)
  @test s2.tol == 1e-4 && s2.gaptol == 0
  @test all(getfield(s2, f) == getfield(s, f) for f in fieldnames(CDSolver) if f ∉ (:tol, :gaptol))
  λs = λmax .* 10 .^ range(0, -1.5; length = 12)
  path = solve_path(ls, NormL1(1.0), λs, CDSolver(; gaptol = 1e-9, gapsafe = true))
  ref = solve_path(ls, NormL1(1.0), λs, CDSolver(; tol = 1e-12))
  for k in eachindex(λs)
    @test norm(path.solutions[k] - ref.solutions[k]) <= 1e-4 * max(1, norm(ref.solutions[k]))
    @test path.values[k] <= ref.values[k] * (1 + 1e-8)
  end
end

@testset "gap-safe screening" begin
  X, y, β = lassodata(; n = 60, p = 400, seed = 11)
  n, p = size(X)
  rng = Xoshiro(12)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  yb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-X * β)))
  logit = (1 / n) * GLMLoss(ProximalBase.Logistic(), yb, X)
  λmax = lambda_max(ls, NormL1(1.0), SparseIterate(p))
  λl = lambda_max(logit, NormL1(1.0), SparseIterate(p))
  # the screened coordinates are zero in a solution to 1e-14
  for (f, g) in ((ls, NormL1(0.3λmax)), (ls, NormL1(0.1λmax, rand(rng, p) .+ 0.5)), (ls, NormL1Nonneg(0.2λmax)),
                 (ls, ElasticNet(0.2λmax, 0.1)), (logit, NormL1(0.3λl)))
    xref = SparseIterate(p)
    solve!(xref, f, g, CDSolver(; tol = 1e-14))
    for s in (CDSolver(; gapsafe = true), CDSolver(; gapsafe = true, gaptol = 1e-10), CDSolver(; gapsafe = true, order = :cyclic))
      x = SparseIterate(p)
      m = cdmodel(f, x)
      PS.initialize!(m, x)
      st = PS._CDState(Float64, x, s.random)
      PS._cd!(x, m, g, st, s)
      screened = findall(!, st.gap.keep)
      # a gap below gaptol at the first check certifies before any screening
      s.gaptol > 0 || @test !isempty(screened)
      @test all(k -> iszero(xref[k]) && iszero(x[k]), screened)
      @test st.candidates === nothing      # restored after the run
      @test norm(Vector(x) - Vector(xref)) <= 1e-5 * max(1, norm(Vector(xref)))
    end
  end
  # groups
  groups = [((k - 1) * 4 + 1):(k * 4) for k in 1:100]
  gg = GroupNormL2(0.25λmax, groups)
  xref = AtomIterate(p, groups)
  solve!(xref, ls, gg, CDSolver(; tol = 1e-14))
  x = AtomIterate(p, groups)
  m = cdmodel(ls, x)
  PS.initialize!(m, x)
  st = PS._CDState(Float64, x, true)
  PS._cd!(x, m, gg, st, CDSolver(; gapsafe = true, tol = 1e-12))
  screened = findall(!, st.gap.keep)
  @test !isempty(screened)
  @test all(k -> !isactive(xref, k) || all(iszero, atom(xref, k)), screened)
  @test norm(Vector(x) - Vector(xref)) <= 1e-5 * max(1, norm(Vector(xref)))
  # screening keeps to the candidates of the strong rule, and restores them
  x = SparseIterate(p)
  m = cdmodel(ls, x)
  PS.initialize!(m, x)
  st = PS._CDState(Float64, x, true)
  base = collect(1:2:p)
  st.candidates = base
  PS._cd!(x, m, NormL1(0.3λmax), st, CDSolver(; gapsafe = true))
  @test st.candidates === base
end

@testset "gaptol in the gradient solvers and ADMM" begin
  X, y, _ = lassodata()
  n, p = size(X)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  λmax = lambda_max(ls, NormL1(1.0), SparseIterate(p))
  g = NormL1(0.02λmax)
  for method in (ProximalGradientDescent(), AccProxGradDescent())
    r = solve!(zeros(p), ls, g, method, ProximalOptions(; maxiter = 50_000, gaptol = 1e-8))
    @test r.gap <= 1e-8
    @test relgap(duality_gap(ls, g, r.minimizer)) <= 1e-8 * (1 + 1e-6)
    r = solve!(SparseIterate(p), ls, g, method, ProximalOptions(; maxiter = 50_000, gaptol = 1e-8, gapevery = 3))
    @test r.gap <= 1e-8
    @test isnan(solve!(zeros(p), ls, g, method).gap)
  end
  # without a gap the other tests apply
  q = QuadraticFunction(X'X / n, -X'y / n)
  r = @test_logs (:warn, r"no duality gap") match_mode = :any solve!(zeros(p), q, g, AccProxGradDescent(), ProximalOptions(; gaptol = 1e-8))
  @test isnan(r.gap) && (r.f_converged || r.x_converged || r.grad_converged)

  # the graphical lasso by ADMM
  rng = Xoshiro(5)
  q = 15
  Ω0 = Matrix(SymTridiagonal(fill(1.0, q), fill(0.45, q - 1)))
  A = randn(rng, 200, q) / cholesky(Ω0).U'
  S = A'A / 200
  W = fill(0.1, q, q); W[diagind(W)] .= 0
  fd, gd = LogDetLoss(S), NormL1(1.0, W)
  r = solve!(Matrix(1.0I, q, q), fd, gd, ADMMSolver(; adaptive = true), ProximalOptions(; maxiter = 10_000, gaptol = 1e-9))
  @test r.converged && r.gap <= 1e-9
  dg = duality_gap(fd, gd, r.minimizer)
  @test relgap(dg) <= 1e-9 * (1 + 1e-6)
  # weak duality at positive definite points, and no dual value off the cone
  for _ in 1:5
    B = randn(rng, q, q)
    Xr = Symmetric(B * B' / q + 0.5I)
    d = duality_gap(fd, gd, Matrix(Xr))
    @test d.gap >= d.primal - dg.primal - 1e-8
  end
  @test duality_gap(fd, gd, -Matrix(1.0I, q, q)).gap == Inf
  # a scaled likelihood, and a penalty on the diagonal
  fw = LogDetLoss(S, 2.0)
  r = solve!(Matrix(1.0I, q, q), fw, NormL1(0.1), ADMMSolver(; adaptive = true), ProximalOptions(; maxiter = 10_000, gaptol = 1e-9))
  @test relgap(duality_gap(fw, NormL1(0.1), r.minimizer)) <= 1e-9 * (1 + 1e-6)
end

@testset "kkt_residual" begin
  X, y, _ = lassodata()
  n, p = size(X)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  λmax = lambda_max(ls, NormL1(1.0), SparseIterate(p))
  q = QuadraticFunction(X'X / n, -X'y / n)
  for (f, g) in ((ls, NormL1(0.1λmax)), (q, NormL1(0.1λmax)), (ls, MCP(0.2λmax)), (q, ElasticNet(0.1λmax, 0.2)))
    x = SparseIterate(p)
    solve!(x, f, g, CDSolver(; tol = 1e-14))
    @test kkt_residual(f, g, x) < 1e-12
    @test kkt_residual(f, g, SparseIterate(p)) > 1e-3
    @test kkt_residual(f, g, SparseIterate(p); relative = false) ≈
          kkt_residual(f, g, SparseIterate(p)) * PS.response_scale(cdmodel(f, SparseIterate(p)), SparseIterate(p))
  end
  Sx, Sy = cov(randn(Xoshiro(15), 50, 10)), cov(randn(Xoshiro(16), 50, 10))
  fk = KronQuadratic(Sx, Sy, Sx - Sy)
  for x in (SymmetricSparseIterate(10), SparseIterate(100))
    solve!(x, fk, NormL1(0.05), CDSolver(; tol = 1e-14))
    @test kkt_residual(fk, NormL1(0.05), x) < 1e-11
  end
  @test_throws ArgumentError kkt_residual(SqrtLeastSquaresLoss(y, X), NormL1(0.1), SparseIterate(p))
end

@testset "Float32" begin
  X, y, _ = lassodata()
  n, p = size(X)
  X32, y32 = Float32.(X), Float32.(y)
  ls = (1f0 / n) * LeastSquaresLoss(y32, X32)
  λmax = lambda_max(ls, NormL1(1f0), SparseIterate(Float32, p))
  x = SparseIterate(Float32, p)
  r = solve!(x, ls, NormL1(0.1f0 * λmax), CDSolver(; gaptol = 1e-5, gapsafe = true))
  @test r.gap isa Float32 && r.gap <= 1f-5
  dg = duality_gap(ls, NormL1(0.1f0 * λmax), x)
  @test dg.gap isa Float32 && relgap(dg) <= 2f-5
end

@testset "allocations" begin
  X, y, β = lassodata()
  n, p = size(X)
  ls = (1 / n) * LeastSquaresLoss(y, X)
  yb = Float64.(rand(Xoshiro(9), n) .< 1 ./ (1 .+ exp.(-X * β)))
  logit = (1 / n) * GLMLoss(ProximalBase.Logistic(), yb, X)
  # a run with the gap allocates nothing once the state holds the gap buffers
  function passes_bytes(x, m, g, s)
    PS.initialize!(m, x)
    st = PS._CDState(Float64, x, s.random)
    PS._cd!(x, m, g, st, s)
    fill!(x, 0.0); PS.initialize!(m, x)
    @allocated PS._cd!(x, m, g, st, s)
  end
  for s in (CDSolver(; gaptol = 1e-9), CDSolver(; gaptol = 1e-9, gapsafe = true),
            CDSolver(; gapsafe = true, order = :cyclic), CDSolver(; gaptol = 1e-9, momentum = false))
    @test passes_bytes(SparseIterate(p), cdmodel(ls, SparseIterate(p)), NormL1(0.1), s) == 0
    @test passes_bytes(SparseIterate(p), cdmodel(logit, SparseIterate(p)), NormL1(0.02), s) == 0
  end
end

end
