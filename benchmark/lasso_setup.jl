# Problems, solvers and accuracy measures for lasso_bench.jl.
#
# Every solver minimizes the same objective
#
#     F(β) = ‖y - Xβ‖² / (2n) + λ‖β‖₁
#
# (no intercept, no standardization). ProximalBase's LeastSquaresLoss is
# ‖y - Xβ‖² / 2, so the ProximalOPT solvers get X/√n and y/√n; the solvers that
# use X'X get A = X'X/n and b = -X'y/n, whose quadratic differs from F only by
# the constant ‖y‖²/(2n).

using LinearAlgebra, Random, Statistics, SparseArrays, Printf
using ProximalBase, CoordinateDescent, ProximalOPT
import GLMNet, Lasso

################################################################################
# problems

struct LassoProblem
  name::String
  desc::String
  X::Matrix{Float64}
  y::Vector{Float64}
  Xs::Matrix{Float64}                  # X / √n
  ys::Vector{Float64}                  # y / √n
  n::Int
  p::Int
  λmax::Float64                        # smallest λ whose solution is zero
  G::Union{Nothing, Matrix{Float64}}   # X'X / n
  mc::Vector{Float64}                  # -X'y / n
  gram_time::Float64                   # seconds to form G
end

"""
Gaussian design with AR(1) correlation ρ between neighboring columns, or with
correlation `common` between every pair of columns (a common factor), columns
scaled so that ‖Xⱼ‖² = n, s nonzero coefficients ±1 at random positions, and
noise with signal-to-noise ratio `snr`. X'X/n is formed when `gram` is true.
"""
function make_problem(name, desc; n, p, s, ρ = 0.0, common = 0.0, snr = 4.0, seed = 1, gram = true)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if ρ > 0
    c = sqrt(1 - ρ^2)
    for j in 2:p
      @views X[:, j] .= ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  if common > 0
    X .= sqrt(1 - common) .* X .+ sqrt(common) .* randn(rng, n)
  end
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  σ = sqrt(sum(abs2, signal) / n / snr)
  y = signal .+ σ .* randn(rng, n)

  mc = -(X' * y) ./ n
  G, gram_time = nothing, NaN
  if gram
    gram_time = @elapsed begin
      G = X' * X
      G ./= n
    end
  end
  LassoProblem(name, desc, X, y, X ./ sqrt(n), y ./ sqrt(n), n, p, maximum(abs, mc), G, mc, gram_time)
end

objective(P::LassoProblem, β, λ) = sum(abs2, P.y - P.X * β) / (2 * P.n) + λ * sum(abs, β)

# largest violation of the optimality conditions ∇f(β) + λ ∂‖β‖₁ ∋ 0
function kkt_residual(P::LassoProblem, β, λ)
  g = P.X' * (P.X * β - P.y) ./ P.n
  r = 0.0
  for j in eachindex(β)
    r = max(r, β[j] == 0 ? max(abs(g[j]) - λ, 0.0) : abs(g[j] + λ * sign(β[j])))
  end
  r
end

# relative suboptimality (F - F*) / F*
relsubopt(F, Fstar) = max((F - Fstar) / abs(Fstar), 0.0)

# coordinates whose zero pattern differs from the reference
support_errors(β, βstar) = count(j -> iszero(β[j]) != iszero(βstar[j]), eachindex(β))

################################################################################
# solvers at one λ: (P, λ, tol) -> dense β, solved from β = 0

struct Solver
  name::String
  package::String
  run::Function
  needs_gram::Bool
end

cd_options(tol; kw...) = CDOptions(; optTol = Float64(tol), maxIter = 1_000_000, kw...)
prox_options(tol) = ProximalOptions(; maxiter = 1_000_000, ftol = tol, xtol = tol, gradtol = tol,
                                    time_limit = 60.0)

function run_cd(P, λ, tol; kw...)
  x = SparseIterate(P.p)
  coordinateDescent!(x, CDLeastSquaresLoss(P.y, P.X), ProxL1(λ), cd_options(tol; kw...))
  Vector(x)
end

function run_cd_gram(P, λ, tol)
  x = SparseIterate(P.p)
  coordinateDescent!(x, CDQuadraticLoss(P.G, P.mc), ProxL1(λ), cd_options(tol; randomize = true))
  Vector(x)
end

function run_prox(P, λ, tol, method; gram = false)
  f = gram ? QuadraticFunction(P.G, P.mc) : LeastSquaresLoss(P.ys, P.Xs)
  x = method isa ActiveAccProxGradDescent ? SparseIterate(P.p) : zeros(P.p)
  solve!(x, f, ProxL1(λ), method, prox_options(tol))
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
  Solver("CD cyclic",               "CoordinateDescent", (P, λ, t) -> run_cd(P, λ, t; randomize = false), false),
  Solver("CD random",               "CoordinateDescent", (P, λ, t) -> run_cd(P, λ, t; randomize = true), false),
  Solver("CD path from λmax",       "CoordinateDescent", (P, λ, t) -> run_cd(P, λ, t; randomize = true, warmStart = false), false),
  Solver("CD X'X/n",                "CoordinateDescent", run_cd_gram, true),
  Solver("ProxGrad",                "ProximalOPT", (P, λ, t) -> run_prox(P, λ, t, ProximalGradientDescent()), false),
  Solver("AccProxGrad",             "ProximalOPT", (P, λ, t) -> run_prox(P, λ, t, AccProxGradDescent()), false),
  Solver("ActiveAccProxGrad",       "ProximalOPT", (P, λ, t) -> run_prox(P, λ, t, ActiveAccProxGradDescent()), false),
  Solver("AccProxGrad X'X/n",       "ProximalOPT", (P, λ, t) -> run_prox(P, λ, t, AccProxGradDescent(); gram = true), true),
  Solver("ActiveAccProxGrad X'X/n", "ProximalOPT", (P, λ, t) -> run_prox(P, λ, t, ActiveAccProxGradDescent(); gram = true), true),
  Solver("glmnet",                  "GLMNet.jl", run_glmnet, false),
  Solver("Lasso.jl",                "Lasso.jl", run_lassojl, false),
]

# each package at its default options
const DEFAULT_RUNS = [
  ("CD lasso()",        (P, λ) -> Vector(lasso(P.X, P.y, λ).x)),
  ("ProxGrad",          (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), ProxL1(λ), ProximalGradientDescent()); x)),
  ("AccProxGrad",       (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), ProxL1(λ), AccProxGradDescent()); x)),
  ("ActiveAccProxGrad", (P, λ) -> (x = zeros(P.p); solve!(x, LeastSquaresLoss(P.ys, P.Xs), ProxL1(λ), ActiveAccProxGradDescent()); x)),
  ("glmnet",            (P, λ) -> convert(Matrix{Float64}, GLMNet.glmnet(P.X, P.y; lambda = [λ], standardize = false, intercept = false).betas)[:, 1]),
  ("Lasso.jl",          (P, λ) -> Vector(Lasso.coef(Lasso.fit(Lasso.LassoPath, P.X, P.y; λ = [λ], standardize = false, intercept = false, maxncoef = P.p))[:, 1])),
]

################################################################################
# solvers along a path of λ values, each warm started from the previous solution:
# (P, λs, tol) -> vector of dense solutions

function path_cd(P, λs, tol)
  path = CoordinateDescent.LassoPath(P.X, P.y, λs, cd_options(tol; randomize = true); standardizeX = false)
  [Vector(b) for b in path.βpath]
end

function path_cd_gram(P, λs, tol)
  x = SparseIterate(P.p)
  f = CDQuadraticLoss(P.G, P.mc)
  [(coordinateDescent!(x, f, ProxL1(λ), cd_options(tol; randomize = true)); Vector(x)) for λ in λs]
end

function path_prox(P, λs, tol, method; gram = false)
  f = gram ? QuadraticFunction(P.G, P.mc) : LeastSquaresLoss(P.ys, P.Xs)
  x = method isa ActiveAccProxGradDescent ? SparseIterate(P.p) : zeros(P.p)
  o = prox_options(tol)
  [(solve!(x, f, ProxL1(λ), method, o); Vector(x)) for λ in λs]
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
  Solver("CD LassoPath",            "CoordinateDescent", path_cd, false),
  Solver("CD X'X/n",                "CoordinateDescent", path_cd_gram, true),
  Solver("ProxGrad",                "ProximalOPT", (P, λs, t) -> path_prox(P, λs, t, ProximalGradientDescent()), false),
  Solver("AccProxGrad",             "ProximalOPT", (P, λs, t) -> path_prox(P, λs, t, AccProxGradDescent()), false),
  Solver("ActiveAccProxGrad",       "ProximalOPT", (P, λs, t) -> path_prox(P, λs, t, ActiveAccProxGradDescent()), false),
  Solver("ActiveAccProxGrad X'X/n", "ProximalOPT", (P, λs, t) -> path_prox(P, λs, t, ActiveAccProxGradDescent(); gram = true), true),
  Solver("glmnet",                  "GLMNet.jl", path_glmnet, false),
  Solver("Lasso.jl",                "Lasso.jl", path_lassojl, false),
]

################################################################################
# timing and reference solutions

"""
Run `f()` at least `minreps` times, and more until the total time reaches
`budget` seconds (at most `maxreps` runs). Returns the first output and the
minimum time: other load on the machine only ever adds time.
"""
function timed(f; budget = 1.0, minreps = 3, maxreps = 7)
  t = @elapsed out = f()
  times = [t]
  while length(times) < minreps || (sum(times) < budget && length(times) < maxreps)
    push!(times, @elapsed f())
  end
  out, minimum(times)
end

# coordinate descent run to optTol = 1e-14, in random order: the order 1, 2, …
# can be very slow when all columns share a common factor
function reference(P, λ)
  β = run_cd(P, λ, 1e-14; randomize = true)
  β, objective(P, β, λ), kkt_residual(P, β, λ)
end

function path_reference(P, λs)
  x = SparseIterate(P.p)
  f = CDLeastSquaresLoss(P.y, P.X)
  [(coordinateDescent!(x, f, ProxL1(λ), cd_options(1e-14; randomize = true)); Vector(x)) for λ in λs]
end

env_summary() = string("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", BLAS threads = ",
                       BLAS.get_num_threads(), ", ", basename(BLAS.get_config().loaded_libs[1].libname))
