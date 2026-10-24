# The graphical lasso
#
#     minimize  tr(SX) - log det X + λ Σᵢ≠ⱼ |Xᵢⱼ|
#
# by ProxNewton (QUIC), by ADMMSolver (the algorithm of CovarianceSelection's
# covsel!) and by GraphicalLassoQUIC.jl, a pure-Julia port of the C QUIC. Each
# run records its time (the fastest of a few) and the relative duality gap of
# its solution, from duality_gap, so that the accuracies compare.
#
#     julia --project=benchmark/quic benchmark/quic/quic_bench.jl [--quick] [--out FILE]

using LinearAlgebra, Random, Printf
using ProximalBase, ProximalSolvers
import GraphicalLassoQUIC

const QUICK = "--quick" in ARGS
const OUT = let i = findfirst(==("--out"), ARGS); i === nothing ? joinpath(@__DIR__, "quic_results.csv") : ARGS[i+1] end

# n samples from N(0, Ω⁻¹) with an AR(1)-like sparse precision: a tridiagonal Ω
# with 0.4 next to the diagonal, plus 0.2 between every fourth pair
function covariance(p, n; seed = 1)
  rng = Xoshiro(seed)
  Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.4, p - 1)))
  for i in 1:p-4
    Ω[i, i + 4] = Ω[i + 4, i] = 0.2
  end
  Ω += max(0.0, 0.1 - eigmin(Symmetric(Ω))) * I
  Z = randn(rng, n, p) / cholesky(Symmetric(Ω)).U'
  Matrix(Symmetric(Z'Z / n))
end

function fastest(run; rounds)
  best, res = Inf, nothing
  for _ in 1:rounds
    GC.gc()
    t = @elapsed r = run()
    t < best && ((best, res) = (t, r))
  end
  best, res
end

relgap(f, g, X) = (d = duality_gap(f, g, X); max(d.gap, 0.0) / abs(d.primal))

function case(io, p, λ; rounds)
  S = covariance(p, 2p; seed = p)
  Λ = fill(λ, p, p); Λ[diagind(Λ)] .= 0
  f, g = LogDetLoss(S), NormL1(1.0, Λ)
  report(label, t, X, extra) = begin
    gap = relgap(f, g, X)
    nz = count(!iszero, X) - p
    @printf "p %5d  λ %.3f  %-26s %9.3f s  gap %.1e  off-diagonal nonzeros %7d  %s\n" p λ label t gap nz extra
    println(io, join((p, λ, label, t, gap, nz, extra), ","))
  end
  for tol in (1e-4, 1e-8)
    t, r = fastest(() -> solve!(SymmetricSparseIterate(p), f, g, ProxNewton(; tol)); rounds)
    report("ProxNewton tol $tol", t, Matrix(r.minimizer), "passes $(r.passes)")
  end
  for (label, opt) in (("ADMM default", ProximalOptions(; maxiter = 5000, time_limit = 120.0)),
                       ("ADMM gaptol 1e-4", ProximalOptions(; maxiter = 5000, gaptol = 1e-4, time_limit = 120.0)))
    t, r = fastest(() -> solve!(Matrix(1.0I, p, p), f, g, ADMMSolver(; adaptive = true), opt); rounds)
    report(label, t, r.minimizer, "iterations $(r.iterations)")
  end
  for tol in (1e-4, 1e-8)
    t, X = fastest(() -> GraphicalLassoQUIC.QUIC(S, Λ; tol); rounds)
    report("GraphicalLassoQUIC tol $tol", t, X, "")
  end
end

function main()
  redirect_stdout(devnull) do
    case(IOBuffer(), 20, 0.2; rounds = 1)        # compile
  end
  cases = QUICK ? [(50, 0.2)] : [(100, 0.2), (100, 0.08), (300, 0.2), (300, 0.08), (600, 0.15)]
  open(OUT, "w") do io
    println(io, "p,lambda,solver,seconds,relgap,offdiag_nonzeros,info")
    for (p, λ) in cases
      case(io, p, λ; rounds = p <= 100 ? 3 : 2)
      flush(io)
    end
  end
  println("wrote ", OUT)
end

main()
