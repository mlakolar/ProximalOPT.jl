# The new code: ProximalOPT's coordinate descent on ProximalBase 0.5, on the
# cases of old/run.jl, under the same keys where both have the case.
#
#     julia --project=benchmark/engine/new -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/engine/new benchmark/engine/new/run.jl [--only REGEX]
#
# Writes benchmark/engine/results/new_<timestamp>.csv and .jls; compare them
# with a run of old/run.jl:
#
#     julia benchmark/engine/compare.jl results/old_<stamp>.csv results/new_<stamp>.csv

using ProximalOPT, ProximalBase
const PO = ProximalOPT
using LinearAlgebra, Random, SparseArrays, SpecialFunctions, Statistics

include(joinpath(@__DIR__, "..", "common.jl"))
BLAS.set_num_threads(1)

const SEED = 20261023
const rec = Recorder("new"; only = only_from_args(ARGS))

cds(; kw...) = CDSolver(; maxpasses = 1_000_000, kw...)
out(x::SparseIterate{T, 1}, r) where {T} = (Vector(x), r.passes, r.converged)
out(x::SymmetricSparseIterate, r) = (vec(Matrix(x)), r.passes, r.converged)
out(x::AtomIterate, r) = (Vector(x), r.passes, r.converged)

println(environment_summary())

################################################################################
# lasso at one λ, from zero

for name in (:A, :B, :C, :D, :E)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  λmax = lasso_λmax(X, y)
  f = (1 / n) * LeastSquaresLoss(y, X)
  for frac in (0.1, 0.02)
    λ = frac * λmax
    prob = "$name λ=$(frac)λmax"
    obj = x -> lasso_objective(X, y, x, λ)
    for order in (:cyclic, :random)
      run!(rec, Case("lasso", prob, "CD $order"); objective = obj) do
        Random.seed!(SEED)
        x = SparseIterate(p)
        out(x, solve!(x, f, NormL1(λ), cds(; order)))
      end
    end
    if frac == 0.02
      run!(rec, Case("lasso", prob, "CD random path from λmax"); objective = obj) do
        Random.seed!(SEED)
        x = SparseIterate(p)
        out(x, solve!(x, f, NormL1(λ), cds(); continuation = 50))
      end
    end
    if name != :D
      A, b = Matrix(Symmetric(X' * X ./ n)), -(X' * y) ./ n
      fq = QuadraticFunction(A, b)
      run!(rec, Case("lasso", prob, "CD X'X/n cyclic"); objective = obj) do
        x = SparseIterate(p)
        out(x, solve!(x, fq, NormL1(λ), cds(; order = :cyclic)))
      end
    end
  end
end

# with an intercept and standardized columns, as lasso(...; intercept = true, standardizeX = true)
let P = design(:A), λ = 0.05 * lasso_λmax(design(:A).X, design(:A).y)
  X, y = P.X, P.y
  n, p = size(X)
  run!(rec, Case("lasso", "A λ=0.05λmax intercept", "lasso() random")) do
    Random.seed!(SEED)
    x = SparseIterate(p)
    m = cdmodel((1 / n) * LeastSquaresLoss(y, X), x; intercept = true)
    xbar = vec(sum(X; dims = 1)) ./ n
    ω = vec(sqrt.(sum(abs2, X .- xbar'; dims = 1) ./ n))
    r = solve!(x, m, NormL1(λ, ω), cds())
    (vcat(PO.intercept(m, x), Vector(x)), r.passes, r.converged)
  end
end

################################################################################
# lasso path: 50 λ from λmax to λmax/100, strong rule

for name in (:A, :B, :C, :D, :E)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  λs = lasso_λmax(X, y) .* 10 .^ range(0, -2; length = 50)
  run!(rec, Case("path", "$name 50λ", "LassoPath random"); budget = 2.0) do
    Random.seed!(SEED)
    path = solve_path((1 / n) * LeastSquaresLoss(y, X), NormL1(1.0), λs, cds())
    (reduce(vcat, Vector.(path.solutions)), sum(path.passes), all(path.converged))
  end
end

################################################################################
# group lasso, groups of 5 columns

for name in (:A, :C)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  groups = consecutive_groups(p, 5)
  λmax = group_λmax(X, y, groups)
  for frac in (0.1, 0.02)
    λ = frac * λmax
    g = GroupNormL2(λ, groups, ones(length(groups)))
    obj = x -> group_objective(X, y, x, λ, groups)
    run!(rec, Case("group", "$name λ=$(frac)λmax", "BCD cyclic"); objective = obj) do
      x = AtomIterate(Float64, p, groups)
      out(x, solve!(x, (1 / n) * LeastSquaresLoss(y, X), g, cds(; order = :cyclic)))
    end
  end
end

################################################################################
# square-root lasso, scaled lasso, heteroscedastic (feasible) lasso

# the loadings Γⱼ = √(∑ᵢ Xᵢⱼ² rᵢ² / n) of the feasible lasso, from the residuals
function loadings(X, r)
  n, p = size(X)
  Γ = zeros(p)
  @inbounds for j in 1:p
    v = 0.0
    @simd for i in 1:n
      v += abs2(X[i, j] * r[i])
    end
    Γ[j] = sqrt(v / n)
  end
  Γ
end

for name in (:A, :C)
  P = design(name)
  X, y = P.X, P.y
  n, p = size(X)
  ω = column_scales(X)
  λmax = maximum(abs.(X' * y) ./ ω) / norm(y)
  for frac in (0.6, 0.4)
    run!(rec, Case("sqrtlasso", "$name λ=$(frac)λmax", "sqrtLasso() random")) do
      Random.seed!(SEED)
      x = SparseIterate(p)
      r = solve!(x, SqrtLeastSquaresLoss(y, X), NormL1(frac * λmax, ω), cds())
      (Vector(x), r.passes, r.converged)
    end
  end
  λ = sqrt(2 * log(p) / n)
  f = (1 / n) * LeastSquaresLoss(y, X)
  # σ̂ = ‖r‖/√n from the residuals of the model; the level is λσ̂
  run!(rec, Case("scaledlasso", "$name λ=√(2log p/n)", "scaledLasso() random")) do
    Random.seed!(SEED)
    σ0 = std(screening_residuals(X, y, 5))
    update = (g, x, m) -> (σ = g.λ0 / λ; σnew = sqrt(sum(abs2, PO.residuals(m)) / n);
                           (NormL1(λ * σnew, ω), abs(σnew - σ) / σ))
    x = SparseIterate(p)
    res = reweight!(x, f, NormL1(λ * σ0, ω), update, cds())
    (vcat(res.next.λ0 / λ, Vector(x)), res.passes, res.converged && res.cd_converged)
  end
  run!(rec, Case("feasiblelasso", "$name λ=√(2log p/n)", "feasibleLasso!() random")) do
    Random.seed!(SEED)
    update = (g, x, m) -> (Γ = loadings(X, PO.residuals(m)); (NormL1(λ, Γ), maximum(abs.(g.w .- Γ)) / maximum(Γ)))
    x = SparseIterate(p)
    res = reweight!(x, f, NormL1(λ, loadings(X, screening_residuals(X, y, 5))), update, cds())
    (Vector(x), res.passes, res.converged && res.cd_converged)
  end
end

################################################################################
# direct difference estimation: the D-trace loss on a SymmetricSparseIterate

for p in (20, 50, 100)
  D = difference_problem(p)
  Sx, Sy = D.Sx, D.Sy
  f = KronQuadratic(Sx, Sy, Sx - Sy)
  λmax = dtrace_λmax(Sx, Sy)
  for frac in (0.5, 0.2)
    λ = frac * λmax
    obj = x -> dtrace_objective(Sx, Sy, reshape(x, p, p), λ)
    for order in (:cyclic, :random)
      run!(rec, Case("dtrace", "p=$p λ=$(frac)λmax", "CD $order"); objective = obj) do
        Random.seed!(SEED)
        x = SymmetricSparseIterate(p)
        out(x, solve!(x, f, NormL1(λ), cds(; order)))
      end
    end
    if frac == 0.2
      run!(rec, Case("dtrace", "p=$p λ=$(frac)λmax", "CD random path from λmax"); objective = obj) do
        Random.seed!(SEED)
        x = SymmetricSparseIterate(p)
        out(x, solve!(x, f, NormL1(λ), cds(); continuation = 50))
      end
    end
  end
end

# the exact refit on the support of a solution, in place of the 1e10 weights
let p = 50
  D = difference_problem(p)
  f = KronQuadratic(D.Sx, D.Sy, D.Sx - D.Sy)
  x0 = SymmetricSparseIterate(p)
  Random.seed!(SEED)
  solve!(x0, f, NormL1(0.2 * dtrace_λmax(D.Sx, D.Sy)), cds())
  obj = v -> dtrace_objective(D.Sx, D.Sy, reshape(v, p, p), 0.0)
  run!(rec, Case("refit", "dtrace p=$p support of λ=0.2λmax", "differencePrecisionRefit random"); objective = obj) do
    x = copy(x0)
    refit!(x, f)
    (vec(Matrix(x)), 0, true)
  end
end

# iterative hard thresholding as projected gradient with backtracking, stopped
# at about the same change of the objective
let p = 50
  D = difference_problem(p)
  f = KronQuadratic(D.Sx, D.Sy, D.Sx - D.Sy)
  s = 2p - 1
  o = ProximalOptions(; maxiter = 1000, ftol = 2e-7, xtol = 1e-14, gradtol = 1e-14)
  obj = v -> dtrace_objective(D.Sx, D.Sy, reshape(v, p, p), 0.0)
  run!(rec, Case("iht", "dtrace p=$p s=$s", "differencePrecisionIHT"); objective = obj) do
    Δ = SymmetricSparseIterate(p)
    r = solve!(Δ, f, IndBallL0(s), ProximalGradientDescent(), o)
    (vec(Matrix(Δ)), r.iterations, PO.converged(r))
  end
end

# a path of 20 λ, each warm started from the previous solution, as DiffPrecTest's
# DTraceValidationSupport solves it
let p = 50
  D = difference_problem(p)
  λs = dtrace_λmax(D.Sx, D.Sy) .* 10 .^ range(0, -1.3; length = 20)
  f = KronQuadratic(D.Sx, D.Sy, D.Sx - D.Sy)
  run!(rec, Case("dtrace", "p=$p 20λ path", "warm starts random"); budget = 2.0) do
    Random.seed!(SEED)
    path = solve_path(f, NormL1(1.0), λs, cds(); x = SymmetricSparseIterate(p), screening = false)
    (reduce(vcat, vec.(Matrix.(path.solutions))), sum(path.passes), all(path.converged))
  end
end

################################################################################
# DiffPrecTest's inverse Hessians at fixed weights: the Kronecker loss on vec Θ,
# and the reduced symmetric loss as a KronQuadratic of a symmetric Θ

for p in (30, 50)
  D = difference_problem(p; n = 500)
  Sx, Sy, X, Y = D.Sx, D.Sy, D.X, D.Y
  λ = 1.01 * (-sqrt(2) * erfcinv(2 * (1 - 0.1 / (p * (p + 1)))))   # DiffPrecTest._penaltyLevel(p)
  ri, ci = 2, 1
  t = (ci - 1) * p + ri
  ωK = kron_weights(Sx, Sy, X, Y, ri, ci); ωK[t] = 0
  E = zeros(p, p); E[ri, ci] = -1
  fK = KronQuadratic(Sx, Sy, E)
  run!(rec, Case("diffprectest", "p=$p", "Kronecker CD only random")) do
    Random.seed!(SEED)
    x = SparseIterate(p * p); x[t] = 1.0
    out(x, solve!(x, fK, NormL1(λ, ωK), cds()))
  end

  # the reduced loss θ'Hθ/2 - θ[j] in svec coordinates θ is ⟨Θ, SxΘSy⟩/2 + ⟨C, Θ⟩
  # of the symmetric Θ, with C = -E_rr on the diagonal and -(E_rc + E_cr)/√2
  # off it; the weight ω of θₖ is ω on Θ[r, r], or ω/√2 on each of Θ[r, c] and
  # Θ[c, r]
  j = ProximalBase.tril_index(p, ri, ci)
  ωS = svec_weights(ωK, p, j)
  W = zeros(p, p)
  C = zeros(p, p)
  k = 0
  for c in 1:p, r in c:p
    k += 1
    W[r, c] = W[c, r] = r == c ? ωS[k] : ωS[k] / sqrt(2)
  end
  ri == ci ? (C[ri, ri] = -1) : (C[ri, ci] = C[ci, ri] = -1 / sqrt(2))
  fS = KronQuadratic(Sx, Sy, C)
  run!(rec, Case("diffprectest", "p=$p", "reduced symmetric Kronecker CD only random")) do
    Random.seed!(SEED)
    x = SymmetricSparseIterate(p)
    ri == ci ? (x[ri, ri] = 1.0) : (x[ri, ci] = 1 / sqrt(2))
    r = solve!(x, fS, NormL1(λ, W), cds())
    # back to svec coordinates
    θ = Float64[]
    for c in 1:p, rr in c:p
      push!(θ, rr == c ? x[rr, c] : sqrt(2) * x[rr, c])
    end
    (θ, r.passes, r.converged)
  end
end

################################################################################
# the proximal gradient solvers on the lasso, and ADMM for the graphical lasso

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
    r = solve!(x, LeastSquaresLoss(ys, Xs), NormL1(λ), AccProxGradDescent(), o)
    (x, r.iterations, PO.converged(r))
  end
  run!(rec, Case("proxgrad", "$name λ=0.1λmax", "ActiveAccProxGrad tol 1e-10"); objective = obj) do
    x = SparseIterate(p)
    r = solve!(x, LeastSquaresLoss(ys, Xs), NormL1(λ), ActiveAccProxGradDescent(), o)
    (Vector(x), r.inner_iterations, r.converged)
  end
end

for p in (50, 100)
  S = Matrix(difference_problem(p).Sy)
  # with the settings of CovSel's covsel!, scaled by the average variance s
  s = tr(S) / p
  covsel = ADMMSolver(; ρ = s^2, α = 1.5, adaptive = true, abstol = 1e-5 / s, abstol_dual = 1e-5 * s, reltol = 1e-3)
  run!(rec, Case("admm", "glasso p=$p λ=0.1", "CovSel covsel!")) do
    Xa = zeros(p, p)
    r = solve!(Xa, LogDetLoss(S), NormL1(0.1), covsel, ProximalOptions(; maxiter = 10_000))
    (vec(Xa), r.iterations, r.converged)
  end
  run!(rec, Case("admm", "glasso p=$p λ=0.1", "ProximalOPT ADMMSolver")) do
    Xa = Matrix(1.0I, p, p)
    r = solve!(Xa, LogDetLoss(S), NormL1(0.1), ADMMSolver(), ProximalOptions(; maxiter = 100_000))
    (vec(Xa), r.iterations, r.converged)
  end
end

println("\nwrote ", save(rec))
