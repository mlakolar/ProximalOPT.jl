# Problems and solver runs for run.jl.
#
# Each problem lists its runs: a solver of ProximalSolvers or of
# ProximalAlgorithms (through adapter.jl) on the same ProximalBase objects,
# started from the same point. A run is a function of a tolerance, returning
# the solution and the number of iterations; `iters(k)` runs exactly k
# iterations, to measure the bytes allocated per iteration.

using LinearAlgebra, Random, SparseArrays, Statistics, Printf, Logging
using ProximalBase, ProximalSolvers, ProximalAlgorithms, ProximalCore
import JuMP, HiGHS
include(joinpath(@__DIR__, "adapter.jl"))
include(joinpath(@__DIR__, "local_functions.jl"))

const PA = ProximalAlgorithms

struct Run
  name::String
  package::String
  run::Function                     # tol -> (x, iterations)
  iters::Union{Nothing, Function}   # k -> x after exactly k iterations
end

struct Problem
  family::String
  name::String
  convex::Bool
  objective::Function               # x -> F(x)
  runs::Vector{Run}
  reference::Union{Nothing, Function}   # () -> (F*, how), for convex problems
end

# ProximalSolvers' gradient solvers, ftol = xtol = gradtol = tol as in the lasso benchmark
function ours(method, f, g, x0, tol; maxiter = 1_000_000, time_limit = TIME_LIMIT[])
  x = copy(x0)
  res = solve!(x, f, g, method, ProximalOptions(; maxiter, ftol = tol, xtol = tol, gradtol = tol, time_limit))
  res.minimizer, res.iterations
end
ours_iters(method, f, g, x0, k) =
  solve!(copy(x0), f, g, method, ProximalOptions(; maxiter = k, ftol = 1e-300, xtol = 1e-300, gradtol = 1e-300,
                                                  allow_f_increases = true)).minimizer

function admm(f, g, x0, tol; maxiter = 1_000_000)
  x = copy(x0)
  res = solve!(x, f, g, ADMMSolver(; adaptive = true, abstol = tol, reltol = tol),
               ProximalOptions(; maxiter, time_limit = TIME_LIMIT[]))
  res.minimizer, res.iterations
end
admm_iters(f, g, x0, k) =
  solve!(copy(x0), f, g, ADMMSolver(; adaptive = true, abstol = 1e-300, reltol = 1e-300), ProximalOptions(; maxiter = k)).minimizer

function cd(f, g, p, tol)
  x = SparseIterate(p)
  res = with_logger(NullLogger()) do
    solve!(x, f, g, CDSolver(; tol, maxpasses = 1_000_000))
  end
  Vector(x), res.passes
end
cd_iters(f, g, p, k) = with_logger(NullLogger()) do
  x = SparseIterate(p)
  solve!(x, f, g, CDSolver(; tol = 0.0, maxpasses = k))
  x
end

# ProximalAlgorithms' forward–backward-type solvers on f + g, from x0. The
# step comes from their backtracking (adaptive) unless gamma is given.
pa_runs(f, g, x0; algs) = [Run(name, "ProximalAlgorithms",
                               tol -> run_pa(alg, tol; x0, f = PASmooth(f), g = PAProx(g), kw...),
                               k -> first(run_pa_iters(alg, k; x0, f = PASmooth(f), g = PAProx(g), kw...)))
                           for (name, alg, kw) in algs]

const FB_ALGS = [("ForwardBackward", PA.ForwardBackward, (;)),
                 ("FastForwardBackward", PA.FastForwardBackward, (;)),
                 ("PANOC", PA.PANOC, (;)),
                 ("ZeroFPR", PA.ZeroFPR, (;)),
                 ("PANOCplus", PA.PANOCplus, (;))]

ours_runs(f, g, x0; methods) = [Run(name, "ProximalSolvers",
                                    tol -> ours(m, f, g, x0, tol),
                                    k -> ours_iters(m, f, g, x0, k))
                                for (name, m) in methods]

const GRAD_METHODS = [("ProxGrad", ProximalGradientDescent()), ("AccProxGrad", AccProxGradDescent())]

# largest eigenvalue of X'X / n, by the power method
function sqopnorm(X; iters = 500)
  v = normalize!(ones(size(X, 2)))
  λ = 0.0
  for _ in 1:iters
    w = X' * (X * v)
    λ = norm(w)
    v = w / λ
  end
  λ
end

################################################################################
# 1. The graphical lasso: LogDetLoss(S) + λ‖X‖₁ off the diagonal, from X = I.
#    Convex, with a gradient that is only locally Lipschitz and f = Inf off the
#    positive definite cone.

function glasso(p; n = 2p, λ = 0.1, seed = 1)
  rng = MersenneTwister(seed)
  Ω0 = Matrix(SymTridiagonal(fill(1.0, p), fill(0.45, p - 1)))
  A = randn(rng, n, p) / cholesky(Ω0).U'                # rows from N(0, Ω0⁻¹)
  S = Matrix(Symmetric(A'A / n))
  W = fill(λ, p, p); W[diagind(W)] .= 0
  f, g = LogDetLoss(S), NormL1(1.0, W)
  X0 = Matrix(1.0I, p, p)
  F(X) = f(X) + g(X)
  runs = vcat(ours_runs(f, g, X0; methods = GRAD_METHODS),
              [Run("ADMM", "ProximalSolvers", tol -> admm(f, g, X0, tol), k -> admm_iters(f, g, X0, k))],
              pa_runs(f, g, X0; algs = FB_ALGS))
  # ADMM to tight tolerances, then the best of the tight runs (see run.jl)
  ref() = (F(first(admm(f, g, X0, 1e-13))), "ADMM, abstol = reltol = 1e-13")
  Problem("graphical lasso", "glasso p=$p n=$n λ=$λ", true, F, runs, ref)
end

################################################################################
# 2. Iterative hard thresholding: ‖y - Xβ‖²/(2n) subject to ‖β‖₀ ≤ s. Nonconvex.

function iht_ls(; n = 500, p = 2000, s = 20, snr = 4.0, seed = 2)
  rng = MersenneTwister(seed)
  X = randn(rng, n, p)
  X ./= sqrt.(sum(abs2, X; dims = 1) ./ n)              # ‖Xⱼ‖² = n
  β0 = zeros(p); β0[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  μ = X * β0
  y = μ + (std(μ) / sqrt(snr)) * randn(rng, n)
  f, g = Scaled(LeastSquaresLoss(y, X), 1 / n), IndBallL0(s)
  L = sqopnorm(X) / n
  x0 = zeros(p)
  F(β) = f(β) + g(β)
  runs = vcat(ours_runs(f, g, x0; methods = GRAD_METHODS),
              pa_runs(f, g, x0; algs = [FB_ALGS[1:4]; ("LiLin", PA.LiLin, (; gamma = 1 / L))]))
  Problem("IHT", "IHT least squares n=$n p=$p s=$s", false, F, runs, nothing)
end

################################################################################
# 3. IHT on the D-trace loss of a symmetric Δ, with at most s off-diagonal
#    entries in its lower triangle. Nonconvex.

function iht_dtrace(; p = 30, n = 300, s = 10, seed = 3)
  rng = MersenneTwister(seed)
  Ωx = Matrix(SymTridiagonal(fill(1.0, p), fill(0.3, p - 1)))
  Ωy = copy(Ωx)
  for _ in 1:s ÷ 2                                       # change s/2 off-diagonal entries
    i, j = rand(rng, 1:p), rand(rng, 1:p)
    i == j && continue
    Ωy[i, j] = Ωy[j, i] = 0.2
  end
  Ωy += (abs(min(eigmin(Ωy), 0.0)) + 0.1) * I
  Σ(Ω) = (A = randn(rng, n, p) / cholesky(Symmetric(Ω)).U'; Matrix(Symmetric(A'A / n)))
  Σx, Σy = Σ(Ωx), Σ(Ωy)
  f, g = SymKron(Σx, Σy, Σx - Σy), SymL0(s)
  L = opnorm(Σx) * opnorm(Σy)
  x0 = zeros(p, p)
  F(Δ) = f(Δ) + g(Δ)
  runs = vcat(ours_runs(f, g, x0; methods = GRAD_METHODS),
              pa_runs(f, g, x0; algs = [FB_ALGS[1:4]; ("LiLin", PA.LiLin, (; gamma = 1 / L))]))
  Problem("IHT", "IHT D-trace p=$p s=$s", false, F, runs, nothing)
end

################################################################################
# 4. Least squares with MCP: ‖y - Xβ‖²/(2n) + Σⱼ MCP(βⱼ; λ, a = 3). Nonconvex;
#    coordinate descent is the reference method.

function ls_mcp(; n = 500, p = 2000, s = 20, snr = 4.0, frac = 0.05, seed = 4)
  rng = MersenneTwister(seed)
  X = randn(rng, n, p)
  X ./= sqrt.(sum(abs2, X; dims = 1) ./ n)
  β0 = zeros(p); β0[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  μ = X * β0
  y = μ + (std(μ) / sqrt(snr)) * randn(rng, n)
  λ = frac * norm(X' * y, Inf) / n
  f, g = Scaled(LeastSquaresLoss(y, X), 1 / n), MCP(λ, 3.0)
  x0 = zeros(p)
  F(β) = f(β) + g(β)
  runs = vcat([Run("CD", "ProximalSolvers", tol -> cd(f, g, p, tol), k -> Vector(cd_iters(f, g, p, k)))],
              ours_runs(f, g, x0; methods = GRAD_METHODS),
              pa_runs(f, g, x0; algs = FB_ALGS[1:4]))
  Problem("MCP", "least squares + MCP n=$n p=$p λ=$(frac)λmax", false, F, runs, nothing)
end

################################################################################
# 5. Quantile regression with ℓ1: (1/n) Σᵢ ρ_τ(yᵢ - xᵢ'β) + λ‖β‖₁. Convex and
#    nonsmooth; a primal–dual problem, against the linear program (HiGHS).

function quantile_lp(X, y, τ, λ; tol = 1e-9)
  n, p = size(X)
  m = JuMP.Model(HiGHS.Optimizer)
  JuMP.set_silent(m)
  JuMP.set_attribute(m, "primal_feasibility_tolerance", tol)
  JuMP.set_attribute(m, "dual_feasibility_tolerance", tol)
  JuMP.@variable(m, bp[1:p] >= 0); JuMP.@variable(m, bm[1:p] >= 0)
  JuMP.@variable(m, u[1:n] >= 0);  JuMP.@variable(m, v[1:n] >= 0)
  JuMP.@constraint(m, X * (bp - bm) + u - v .== y)
  JuMP.@objective(m, Min, (τ * sum(u) + (1 - τ) * sum(v)) / n + λ * (sum(bp) + sum(bm)))
  JuMP.optimize!(m)
  JuMP.value.(bp) - JuMP.value.(bm), JuMP.simplex_iterations(m)
end

function quantile_l1(; n = 500, p = 100, s = 10, τ = 0.5, frac = 0.1, seed = 5)
  rng = MersenneTwister(seed)
  X = randn(rng, n, p)
  X ./= sqrt.(sum(abs2, X; dims = 1) ./ n)
  β0 = zeros(p); β0[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  y = X * β0 + randn(rng, n) ./ sqrt.(vec(sum(abs2, randn(rng, n, 3); dims = 2)) ./ 3)   # t₃ noise
  ψ = [yi < 0 ? τ - 1 : τ for yi in y]                 # a subgradient of the loss at β = 0
  λ = frac * norm(X' * ψ, Inf) / n
  h, g = CheckResidual(y, τ, 1 / n), NormL1(λ)
  F(β) = h(X * β) + g(β)
  nL = opnorm(X)
  cp(tol) = (sol = run_pa(PA.ChambollePock, tol; x0 = zeros(p), y0 = zeros(n), g = PAProx(g), h, L = X,
                          gamma = (1 / nL, 0.99 / nL)); (first(sol[1]), sol[2]))
  cp_iters(k) = first(first(run_pa_iters(PA.ChambollePock, k; x0 = zeros(p), y0 = zeros(n), g = PAProx(g), h,
                                         L = X, gamma = (1 / nL, 0.99 / nL))))
  runs = [Run("ChambollePock", "ProximalAlgorithms", cp, cp_iters),
          Run("HiGHS simplex", "JuMP + HiGHS", tol -> quantile_lp(X, y, τ, λ; tol = max(tol, 1e-10)), nothing)]
  ref() = (F(first(quantile_lp(X, y, τ, λ; tol = 1e-10))), "HiGHS, feasibility tolerances 1e-10")
  Problem("quantile regression", "quantile ℓ1 n=$n p=$p τ=$τ", true, F, runs, ref)
end

################################################################################
# 6. 1-D total variation denoising: ‖y - β‖²/2 + λ‖Dβ‖₁, D the differences.
#    The primal–dual solvers, against AccProxGradDescent on the dual
#    min ‖y - D'u‖²/2 subject to ‖u‖∞ ≤ λ, with β = y - D'u.

function tv_denoise(; n = 1000, λ = 1.0, jumps = n ÷ 100, seed = 6)
  rng = MersenneTwister(seed)
  cuts = sort(randperm(rng, n - 1)[1:jumps])
  signal = zeros(n); level = 0.0; prev = 1
  for c in [cuts; n]
    level += randn(rng)
    signal[prev:c] .= level
    prev = c + 1
  end
  y = signal + 0.5 * randn(rng, n)
  D = spdiagm(n - 1, n, 0 => -ones(n - 1), 1 => ones(n - 1))
  Dt = sparse(D')
  F(β) = sum(abs2, y - β) / 2 + λ * norm(D * β, 1)
  nD = 2.0                                              # ‖D‖₂ < 2
  primal(u) = y - Dt * u
  dual(tol) = (sol = ours(AccProxGradDescent(), LeastSquaresLoss(y, Dt), IndBallLinf(λ), zeros(n - 1), tol);
               (primal(sol[1]), sol[2]))
  dual_iters(k) = primal(ours_iters(AccProxGradDescent(), LeastSquaresLoss(y, Dt), IndBallLinf(λ), zeros(n - 1), k))
  cp(tol) = (sol = run_pa(PA.ChambollePock, tol; x0 = zeros(n), y0 = zeros(n - 1), g = PAProx(SqrNormL2(1.0, y)),
                          h = PAProx(NormL1(λ)), L = D, gamma = (1 / nD, 0.99 / nD)); (first(sol[1]), sol[2]))
  cp_iters(k) = first(first(run_pa_iters(PA.ChambollePock, k; x0 = zeros(n), y0 = zeros(n - 1),
                                         g = PAProx(SqrNormL2(1.0, y)), h = PAProx(NormL1(λ)), L = D,
                                         gamma = (1 / nD, 0.99 / nD))))
  vc(tol) = (sol = run_pa(PA.VuCondat, tol; x0 = zeros(n), y0 = zeros(n - 1), f = PASmooth(SqrNormL2(1.0, y)),
                          beta_f = 1.0, h = PAProx(NormL1(λ)), L = D, gamma = (1 / (0.5 + nD), 0.99 / nD));
             (first(sol[1]), sol[2]))
  vc_iters(k) = first(first(run_pa_iters(PA.VuCondat, k; x0 = zeros(n), y0 = zeros(n - 1),
                                         f = PASmooth(SqrNormL2(1.0, y)), beta_f = 1.0, h = PAProx(NormL1(λ)),
                                         L = D, gamma = (1 / (0.5 + nD), 0.99 / nD))))
  runs = [Run("AccProxGrad on the dual", "ProximalSolvers", dual, dual_iters),
          Run("ChambollePock", "ProximalAlgorithms", cp, cp_iters),
          Run("VuCondat", "ProximalAlgorithms", vc, vc_iters)]
  ref() = (F(first(dual(1e-15))), "AccProxGradDescent on the dual, tol = 1e-15")
  Problem("total variation", "TV denoising n=$n λ=$λ", true, F, runs, ref)
end
