# The baseline: the coordinate descent and solver code of CovSel, DiffPrecTest
# and ProximalOPT 0.1 as it was before ProximalOPT took it over, on
# CoordinateDescent 0.5.4 and ProximalBase 0.4. CoordinateDescent's own
# estimators run in old-cd/, on its last release.
#
#     julia --project=benchmark/engine/old -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/engine/old benchmark/engine/old/run.jl [--only REGEX]
#
# Writes benchmark/engine/results/old_<timestamp>.csv and .jls; compare.jl
# compares them, with the runs of old-cd/run.jl, with a run of new/run.jl.
# Random coordinate orders draw from the global random number generator, which
# each run seeds, so that a run is reproducible.

using CoordinateDescent, ProximalBase, ProximalOPT, CovSel, DiffPrecTest
using LinearAlgebra, Random, SparseArrays, Statistics

include(joinpath(@__DIR__, "..", "common.jl"))
BLAS.set_num_threads(1)

const SEED = 20261023
const rec = Recorder("old"; only = only_from_args(ARGS))

cdopts(; kw...) = CDOptions(; maxIter = 1_000_000, kw...)
info_out(x, info::CDInfo) = (x, info.iterations, info.converged)

println(environment_summary())

################################################################################
# direct difference estimation (CovSel's D-trace loss on a SymmetricSparseIterate)

for p in (20, 50, 100)
  D = difference_problem(p)
  Sx, Sy = D.Sx, D.Sy
  λmax = dtrace_λmax(Sx, Sy)
  for frac in (0.5, 0.2)
    λ = frac * λmax
    obj = x -> dtrace_objective(Sx, Sy, reshape(x, p, p), λ)
    for (order, randomize) in (("cyclic", false), ("random", true))
      run!(rec, Case("dtrace", "p=$p λ=$(frac)λmax", "CD $order"); objective = obj) do
        Random.seed!(SEED)
        x, info = SymmetricSparseIterate(p), CDInfo()
        coordinateDescent!(x, CovSel.CDDirectDifferenceLoss(Sx, Sy), ProxL1(λ), cdopts(; randomize), info)
        info_out(vec(Matrix(x)), info)
      end
    end
    if frac == 0.2
      run!(rec, Case("dtrace", "p=$p λ=$(frac)λmax", "CD random path from λmax"); objective = obj) do
        Random.seed!(SEED)
        x, info = SymmetricSparseIterate(p), CDInfo()
        coordinateDescent!(x, CovSel.CDDirectDifferenceLoss(Sx, Sy), ProxL1(λ), cdopts(; warmStart = false), info)
        info_out(vec(Matrix(x)), info)
      end
    end
  end
end

# the refit on the support of a solution, by coordinate descent with the
# entries off the support penalized by 1e10, as CovSel's differencePrecisionRefit
let p = 50
  D = difference_problem(p)
  x = SymmetricSparseIterate(p)
  Random.seed!(SEED)
  coordinateDescent!(x, CovSel.CDDirectDifferenceLoss(D.Sx, D.Sy), ProxL1(0.2 * dtrace_λmax(D.Sx, D.Sy)), cdopts())
  S = [CartesianIndex(ProximalBase.ind2subLowerTriangular(p, k)) for k in 1:div(p * (p + 1), 2) if !iszero(x[ProximalBase.ind2subLowerTriangular(p, k)...])]
  obj = v -> dtrace_objective(D.Sx, D.Sy, reshape(v, p, p), 0.0)
  run!(rec, Case("refit", "dtrace p=$p support of λ=0.2λmax", "differencePrecisionRefit random"); objective = obj) do
    Random.seed!(SEED)
    (vec(Matrix(CovSel.differencePrecisionRefit(D.Sx, D.Sy, S))), -1, true)
  end
end

# iterative hard thresholding: at most 2p - 1 nonzero entries in the lower
# triangle, a fixed step 1/(‖Sx‖‖Sy‖), as CovSel's benchmark
let p = 50
  D = difference_problem(p)
  s, η = 2p - 1, 1 / (opnorm(Matrix(D.Sx)) * opnorm(Matrix(D.Sy)))
  obj = v -> dtrace_objective(D.Sx, D.Sy, reshape(v, p, p), 0.0)
  run!(rec, Case("iht", "dtrace p=$p s=$s", "differencePrecisionIHT"); objective = obj) do
    Δ, fvals = CovSel.differencePrecisionIHT(D.Sx, D.Sy, η, s)
    (vec(Matrix(Δ)), length(fvals) - 1, length(fvals) - 1 < 1000)
  end
end

# a path of 20 λ, each warm started from the previous solution, as DiffPrecTest's
# DTraceValidationSupport solves it (without its validation losses)
let p = 50
  D = difference_problem(p)
  λs = dtrace_λmax(D.Sx, D.Sy) .* 10 .^ range(0, -1.3; length = 20)
  run!(rec, Case("dtrace", "p=$p 20λ path", "warm starts random"); budget = 2.0) do
    Random.seed!(SEED)
    f = CovSel.CDDirectDifferenceLoss(D.Sx, D.Sy)
    x, info = SymmetricSparseIterate(p), CDInfo()
    sols, passes, conv = Vector{Float64}[], 0, true
    for λ in λs
      coordinateDescent!(x, f, ProxL1(λ), cdopts(), info)
      push!(sols, vec(Matrix(x))); passes += info.iterations; conv &= info.converged
    end
    (reduce(vcat, sols), passes, conv)
  end
end

################################################################################
# DiffPrecTest: the two-stage direct difference estimate and the rows of the
# inverse Hessians, at its penalty level _penaltyLevel(p), with weights from the
# data. The "CD only" cases time the coordinate descent of the first stage at
# fixed weights, which the weights computation would otherwise dominate.

for p in (30, 50)
  D = difference_problem(p; n = 500)
  Sx, Sy, X, Y = D.Sx, D.Sy, D.X, D.Y
  λ = DiffPrecTest._penaltyLevel(p)
  ri, ci = 2, 1
  run!(rec, Case("diffprectest", "p=$p", "diffEstimation random"); budget = 2.0) do
    Random.seed!(SEED)
    (vec(Matrix(DiffPrecTest.diffEstimation(Sx, Sy, X, Y, λ))), -1, true)
  end
  run!(rec, Case("diffprectest", "p=$p", "invAsymHessianEstimation random"); budget = 2.0) do
    Random.seed!(SEED)
    (Vector(DiffPrecTest.invAsymHessianEstimation(Sx, Sy, ri, ci, X, Y, λ)), -1, true)
  end
  j = ProximalBase.sub2indLowerTriangular(p, ri, ci)
  run!(rec, Case("diffprectest", "p=$p", "invQSymHessian random"); budget = 2.0) do
    Random.seed!(SEED)
    (Vector(DiffPrecTest.invQSymHessian(Sx, Sy, j, X, Y, λ)), -1, true)
  end

  # first stages at fixed weights
  fK = DiffPrecTest.CDInverseKroneckerLoss(Sx, Sy, ri, ci)
  t = (ci - 1) * p + ri
  x0 = SparseIterate(p * p); x0[t] = 1.0
  CoordinateDescent.initialize!(fK, x0)
  ωK = zeros(p * p); DiffPrecTest._computeVar!(ωK, fK, X, Y, x0)
  ωK ≈ kron_weights(Sx, Sy, X, Y, ri, ci) || error("kron_weights does not match DiffPrecTest._computeVar!")
  ωK = kron_weights(Sx, Sy, X, Y, ri, ci); ωK[t] = 0
  run!(rec, Case("diffprectest", "p=$p", "Kronecker CD only random")) do
    Random.seed!(SEED)
    x = SparseIterate(p * p); x[t] = 1.0
    info = CDInfo()
    coordinateDescent!(x, DiffPrecTest.CDInverseKroneckerLoss(Sx, Sy, ri, ci), ProxL1(λ, ωK), cdopts(), info)
    info_out(Vector(x), info)
  end
  fS = DiffPrecTest.CDInverseReducedSymKroneckerLoss(Sx, Sy, j)
  y0 = SparseIterate(div(p * (p + 1), 2)); y0[j] = 1.0
  CoordinateDescent.initialize!(fS, y0)
  # the weights of invQSymHessian are standard errors; any positive weights do
  # for timing, so take ωK at the matching lower-triangle entries
  ωS = svec_weights(ωK, p, j)
  run!(rec, Case("diffprectest", "p=$p", "reduced symmetric Kronecker CD only random")) do
    Random.seed!(SEED)
    x = SparseIterate(div(p * (p + 1), 2)); x[j] = 1.0
    info = CDInfo()
    coordinateDescent!(x, DiffPrecTest.CDInverseReducedSymKroneckerLoss(Sx, Sy, j), ProxL1(λ, ωS), cdopts(), info)
    info_out(Vector(x), info)
  end
end

################################################################################
# ProximalOPT 0.1: the proximal gradient solvers on the lasso, and ADMM for the
# graphical lasso

for name in (:A, :E)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  Xs, ys = X ./ sqrt(n), y ./ sqrt(n)
  λ = 0.1 * lasso_λmax(X, y)
  o = ProximalOptions(; maxiter = 1_000_000, ftol = 1e-10, xtol = 1e-10, gradtol = 1e-10, time_limit = 60.0)
  obj = x -> lasso_objective(X, y, x, λ)
  run!(rec, Case("proxgrad", "$name λ=0.1λmax", "AccProxGrad tol 1e-10"); objective = obj) do
    x = zeros(p)
    r = solve!(x, LeastSquaresLoss(ys, Xs), ProxL1(λ), AccProxGradDescent(), o)
    (x, r.iterations, ProximalOPT.converged(r))
  end
  run!(rec, Case("proxgrad", "$name λ=0.1λmax", "ActiveAccProxGrad tol 1e-10"); objective = obj) do
    x = SparseIterate(p)
    r = solve!(x, LeastSquaresLoss(ys, Xs), ProxL1(λ), ActiveAccProxGradDescent(), o)
    (Vector(x), r.inner_iterations, r.converged)
  end
end

for p in (50, 100)
  S = Matrix(difference_problem(p).Sy)
  run!(rec, Case("admm", "glasso p=$p λ=0.1", "CovSel covsel!")) do
    Xa, Z, U = zeros(p, p), zeros(p, p), zeros(p, p)
    info = ADMMInfo()
    covsel!(Xa, Z, U, S, 0.1; info)
    (vec(Z), info.iterations, info.converged)
  end
  run!(rec, Case("admm", "glasso p=$p λ=0.1", "ProximalOPT ADMMSolver")) do
    Xa = Matrix(1.0I, p, p)
    r = solve!(Xa, ProxGaussLikelihood(Symmetric(S)), ProxL1(0.1), ADMMSolver(),
               ProximalOptions(; maxiter = 100_000))
    (vec(Xa), r.iterations, r.converged)
  end
end

println("\nwrote ", save(rec))
