# The baseline of CoordinateDescent's own estimators: CoordinateDescent 0.6.0,
# its last release, which uses momentum by default with the random order, on
# ProximalBase 0.4. CovSel and DiffPrecTest allow only CoordinateDescent 0.4 and
# 0.5, so their cases run in old/, on CoordinateDescent 0.5.4.
#
#     julia --project=benchmark/engine/old-cd -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/engine/old-cd benchmark/engine/old-cd/run.jl [--only REGEX]
#
# Writes benchmark/engine/results/old-cd_<timestamp>.csv and .jls; compare.jl
# takes them, with the runs of old/run.jl, as the old side. Random coordinate
# orders draw from the global random number generator, which each run seeds,
# so that a run is reproducible.

using CoordinateDescent, ProximalBase
using LinearAlgebra, Random, SparseArrays, Statistics

include(joinpath(@__DIR__, "..", "common.jl"))
BLAS.set_num_threads(1)

const SEED = 20261023
const rec = Recorder("old-cd"; only = only_from_args(ARGS))

cdopts(; kw...) = CDOptions(; maxIter = 1_000_000, kw...)
info_out(x, info::CDInfo) = (x, info.iterations, info.converged)

println(environment_summary())

################################################################################
# lasso: CDLeastSquaresLoss at one λ, from zero

for name in (:A, :B, :C, :D, :E)
  P = design(name)
  X, y = P.X, P.y
  p = size(X, 2)
  λmax = lasso_λmax(X, y)
  for frac in (0.1, 0.02)
    λ = frac * λmax
    prob = "$name λ=$(frac)λmax"
    obj = x -> lasso_objective(X, y, x, λ)
    for (order, randomize) in (("cyclic", false), ("random", true))
      run!(rec, Case("lasso", prob, "CD $order"); objective = obj) do
        Random.seed!(SEED)
        x, info = SparseIterate(p), CDInfo()
        coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), cdopts(; randomize), info)
        info_out(Vector(x), info)
      end
    end
    # the path of 50 penalty levels from λmax of warmStart = false
    if frac == 0.02
      run!(rec, Case("lasso", prob, "CD random path from λmax"); objective = obj) do
        Random.seed!(SEED)
        x, info = SparseIterate(p), CDInfo()
        coordinateDescent!(x, CDLeastSquaresLoss(y, X), ProxL1(λ), cdopts(; warmStart = false), info)
        info_out(Vector(x), info)
      end
    end
    # the quadratic X'X/n, -X'y/n, when it fits in memory
    if name != :D
      A, b = Symmetric(X' * X ./ size(X, 1)), -(X' * y) ./ size(X, 1)
      A = Matrix(A)
      run!(rec, Case("lasso", prob, "CD X'X/n cyclic"); objective = obj) do
        x, info = SparseIterate(p), CDInfo()
        coordinateDescent!(x, CDQuadraticLoss(A, b), ProxL1(λ), cdopts(; randomize = false), info)
        info_out(Vector(x), info)
      end
    end
  end
end

# with an intercept and standardized columns, through lasso()
let P = design(:A), λ = 0.05 * lasso_λmax(design(:A).X, design(:A).y)
  X, y = P.X, P.y
  run!(rec, Case("lasso", "A λ=0.05λmax intercept", "lasso() random")) do
    Random.seed!(SEED)
    s = lasso(X, y, λ; intercept = true, standardizeX = true)
    (vcat(s.intercept, Vector(s.x)), s.iterations, s.converged)
  end
end

################################################################################
# lasso path: LassoPath with 50 λ from λmax to λmax/100, strong rule

for name in (:A, :B, :C, :D, :E)
  P = design(name)
  X, y = P.X, P.y
  λs = lasso_λmax(X, y) .* 10 .^ range(0, -2; length = 50)
  run!(rec, Case("path", "$name 50λ", "LassoPath random"); budget = 2.0) do
    Random.seed!(SEED)
    path = CoordinateDescent.LassoPath(X, y, λs, cdopts(); standardizeX = false)
    (reduce(vcat, Vector.(path.βpath)), sum(path.iterations), all(path.converged))
  end
end

################################################################################
# group lasso: block coordinate descent on an AtomIterate, groups of 5 columns

for name in (:A, :C)
  P = design(name)
  X, y = P.X, P.y
  p = size(X, 2)
  groups = consecutive_groups(p, 5)
  λmax = group_λmax(X, y, groups)
  for frac in (0.1, 0.02)
    λ = frac * λmax
    g = ProxGroupL2(λ, groups, ones(length(groups)))
    obj = x -> group_objective(X, y, x, λ, groups)
    run!(rec, Case("group", "$name λ=$(frac)λmax", "BCD cyclic"); objective = obj) do
      x, info = AtomIterate(p, groups), CDInfo()
      coordinateDescent!(x, CDLeastSquaresLoss(y, X), g, cdopts(; randomize = false), info)
      info_out(Vector(x), info)
    end
  end
end

################################################################################
# square-root lasso, scaled lasso, heteroscedastic (feasible) lasso

for name in (:A, :C)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  ω = column_scales(X)
  # sqrt lasso: λmax = maxⱼ |Xⱼ'y| / (ωⱼ ‖y‖)
  λmax = maximum(abs.(X' * y) ./ ω) / norm(y)
  for frac in (0.6, 0.4)
    run!(rec, Case("sqrtlasso", "$name λ=$(frac)λmax", "sqrtLasso() random")) do
      Random.seed!(SEED)
      s = sqrtLasso(X, y, frac * λmax)
      (Vector(s.x), s.iterations, s.converged)
    end
  end
  λ = sqrt(2 * log(p) / n)
  run!(rec, Case("scaledlasso", "$name λ=√(2log p/n)", "scaledLasso() random")) do
    Random.seed!(SEED)
    s = scaledLasso(X, y, λ, ω)
    (vcat(s.σ, Vector(s.x)), s.iterations, s.converged)
  end
  run!(rec, Case("feasiblelasso", "$name λ=√(2log p/n)", "feasibleLasso!() random")) do
    Random.seed!(SEED)
    s = feasibleLasso!(SparseIterate(p), X, y, λ)
    (Vector(s.x), s.iterations, s.converged)
  end
end

################################################################################
# local polynomial lasso (CDWeightedLSLoss), as in CoordinateDescent's
# benchmark/locpoly_bench.jl: n = 500, 2 varying and 48 irrelevant coefficients

let n = 500, s = 2, p = 50, degree = 2
  V = varying_coefficient_data(n, s)
  X = [V.X randn(Xoshiro(2), n, p - s)]
  zgrid = collect(range(0.025, 0.975; length = 20))
  run!(rec, Case("locpoly", "n=500 p=50 degree=2 20 points", "locpolyl1() random")) do
    Random.seed!(SEED)
    B, _ = locpolyl1(X, V.Z, V.Y, zgrid, degree, GaussianKernel(0.2), 0.05, false, cdopts(; optTol = 1e-8))
    (vec(Matrix(B)), -1, true)
  end
end

println("\nwrote ", save(rec))
