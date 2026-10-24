# Problems, solvers and accuracy measures for lasso_bench.jl.
#
# Every solver minimizes the same objective
#
#     F(β) = ‖y - Xβ‖² / (2n) + λ‖β‖₁
#
# (no intercept, no standardization). ProximalBase's LeastSquaresLoss is
# ‖y - Xβ‖² / 2, so the ProximalSolvers solvers get X/√n and y/√n; the solvers that
# use X'X get A = X'X/n and b = -X'y/n, whose quadratic differs from F only by
# the constant ‖y‖²/(2n).

using LinearAlgebra, Random, Statistics, SparseArrays, Printf
using ProximalBase, ProximalSolvers
using Serialization

include(joinpath(@__DIR__, "lasso_common.jl"))
include(joinpath(@__DIR__, "lasso_legacy_client.jl"))
import GLMNet, Lasso

################################################################################
# solvers at one λ: (P, λ, tol) -> dense β, solved from β = 0

struct Solver{F}
  name::String
  package::String
  run::F
  needs_gram::Bool
end

prox_options(tol) = ProximalOptions(; maxiter = 1_000_000, ftol = tol, xtol = tol, gradtol = tol,
                                    time_limit = 60.0)

function run_prox(P, λ, tol, method; gram = false)
  f = gram ? QuadraticFunction(P.G, P.mc) : LeastSquaresLoss(P.ys, P.Xs)
  x = method isa ActiveAccProxGradDescent ? SparseIterate(P.p) : zeros(P.p)
  solve!(x, f, NormL1(λ), method, prox_options(tol))
  x isa SparseIterate ? Vector(x) : x
end

# glmnet's tol bounds squared coefficient changes, so it needs a smaller tol
# than the other solvers for the same accuracy
function run_glmnet(P, λ, tol)
  path = GLMNet.glmnet(P.X, P.y; lambda = [λ], standardize = false, intercept = false, tol = tol)
  convert(Matrix{Float64}, path.betas)[:, 1]
end

function run_lassojl(P, λ, tol)
  path = Lasso.fit(Lasso.LassoPath, P.X, P.y; λ = [λ], standardize = false, intercept = false,
                   cd_tol = tol, cd_maxiter = 1_000_000, stopearly = false, maxncoef = P.p)
  Vector(Lasso.coef(path)[:, 1])
end

const SOLVERS = [
  Solver("CD cyclic",               "CoordinateDescent", LegacyRun(:cyclic), false),
  Solver("CD random",               "CoordinateDescent", LegacyRun(:random), false),
  Solver("CD path from λmax",       "CoordinateDescent", LegacyRun(:continuation), false),
  Solver("CD X'X/n",                "CoordinateDescent", LegacyRun(:gram), true),
  Solver("ProxGrad",                "ProximalSolvers", (P, λ, t) -> run_prox(P, λ, t, ProximalGradientDescent()), false),
  Solver("AccProxGrad",             "ProximalSolvers", (P, λ, t) -> run_prox(P, λ, t, AccProxGradDescent()), false),
  Solver("ActiveAccProxGrad",       "ProximalSolvers", (P, λ, t) -> run_prox(P, λ, t, ActiveAccProxGradDescent()), false),
  Solver("AccProxGrad X'X/n",       "ProximalSolvers", (P, λ, t) -> run_prox(P, λ, t, AccProxGradDescent(); gram = true), true),
  Solver("ActiveAccProxGrad X'X/n", "ProximalSolvers", (P, λ, t) -> run_prox(P, λ, t, ActiveAccProxGradDescent(); gram = true), true),
  Solver("glmnet",                  "GLMNet.jl", run_glmnet, false),
  Solver("Lasso.jl",                "Lasso.jl", run_lassojl, false),
]

# each package at its default options
const DEFAULT_RUNS = [
  ("CD lasso()",        LegacyRun(:default)),
  ("ProxGrad",          (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), NormL1(λ), ProximalGradientDescent()); x)),
  ("AccProxGrad",       (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), NormL1(λ), AccProxGradDescent()); x)),
  ("ActiveAccProxGrad", (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), NormL1(λ), ActiveAccProxGradDescent()); x)),
  ("glmnet",            (P, λ) -> convert(Matrix{Float64}, GLMNet.glmnet(P.X, P.y; lambda = [λ], standardize = false, intercept = false).betas)[:, 1]),
  ("Lasso.jl",          (P, λ) -> Vector(Lasso.coef(Lasso.fit(Lasso.LassoPath, P.X, P.y; λ = [λ], standardize = false, intercept = false, maxncoef = P.p))[:, 1])),
]

################################################################################
# solvers along a path of λ values, each warm started from the previous solution:
# (P, λs, tol) -> vector of dense solutions

function path_prox(P, λs, tol, method; gram = false)
  f = gram ? QuadraticFunction(P.G, P.mc) : LeastSquaresLoss(P.ys, P.Xs)
  x = method isa ActiveAccProxGradDescent ? SparseIterate(P.p) : zeros(P.p)
  o = prox_options(tol)
  [(solve!(x, f, NormL1(λ), method, o); Vector(x)) for λ in λs]
end

function path_glmnet(P, λs, tol)
  path = GLMNet.glmnet(P.X, P.y; lambda = λs, standardize = false, intercept = false, tol = tol)
  B = convert(Matrix{Float64}, path.betas)
  [B[:, k] for k in axes(B, 2)]
end

function path_lassojl(P, λs, tol)
  path = Lasso.fit(Lasso.LassoPath, P.X, P.y; λ = λs, standardize = false, intercept = false,
                   cd_tol = tol, cd_maxiter = 1_000_000, stopearly = false, maxncoef = P.p)
  B = Matrix(Lasso.coef(path))
  [B[:, k] for k in axes(B, 2)]
end

const PATH_SOLVERS = [
  Solver("CD LassoPath",            "CoordinateDescent", LegacyRun(:path), false),
  Solver("CD X'X/n",                "CoordinateDescent", LegacyRun(:path_gram), true),
  Solver("ProxGrad",                "ProximalSolvers", (P, λs, t) -> path_prox(P, λs, t, ProximalGradientDescent()), false),
  Solver("AccProxGrad",             "ProximalSolvers", (P, λs, t) -> path_prox(P, λs, t, AccProxGradDescent()), false),
  Solver("ActiveAccProxGrad",       "ProximalSolvers", (P, λs, t) -> path_prox(P, λs, t, ActiveAccProxGradDescent()), false),
  Solver("ActiveAccProxGrad X'X/n", "ProximalSolvers", (P, λs, t) -> path_prox(P, λs, t, ActiveAccProxGradDescent(); gram = true), true),
  Solver("glmnet",                  "GLMNet.jl", path_glmnet, false),
  Solver("Lasso.jl",                "Lasso.jl", path_lassojl, false),
]

################################################################################
# timing and reference solutions

# Reference solutions come from the same isolated historical implementation.
function reference(P, λ)
  β = LegacyRun(:random)(P, λ, 1e-14)
  β, objective(P, β, λ), lasso_kkt(P, β, λ)
end

path_reference(P, λs) = LegacyRun(:path_reference)(P, λs, 1e-14)
