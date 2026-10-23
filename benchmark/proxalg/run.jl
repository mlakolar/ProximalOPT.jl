# ProximalAlgorithms.jl against ProximalSolvers on the problems of issues #23
# (primal–dual), #24 (nonconvex) and #25 (PANOC, ZeroFPR, PANOCplus): measure
# before writing new solvers. See README.md.
#
#     julia --project=benchmark/proxalg -e 'using Pkg; Pkg.instantiate()'
#     julia --project=benchmark/proxalg benchmark/proxalg/run.jl --quick   # a few minutes
#     julia --project=benchmark/proxalg benchmark/proxalg/run.jl           # about an hour
#     julia --project=benchmark/proxalg benchmark/proxalg/summary.jl
#
# 1. Each solver runs over a sweep of its tolerance, from the same start, and
#    each run records its time, iterations and objective. For a convex problem
#    this gives the time to a relative suboptimality (F - F*)/|F*|; for a
#    nonconvex one, the objective each solver reaches and how fast.
# 2. Bytes and time per iteration, from runs of exactly k₁ and k₂ iterations.
# 3. The graphical lasso from an initial step that leaves the positive definite
#    cone, where LogDetLoss is Inf: which solvers recover.

include(joinpath(@__DIR__, "problems.jl"))
BLAS.set_num_threads(1)

const QUICK = "--quick" in ARGS
const OUT = get(ENV, "PROXALG_OUT", joinpath(@__DIR__, "results"))
const TOLS = QUICK ? 10.0 .^ (-2:-2:-10) : 10.0 .^ (-1:-1:-12)
const RUN_CAP = QUICK ? 5.0 : 30.0       # stop a sweep after a run this slow (seconds)
const SOLVER_CAP = QUICK ? 15.0 : 150.0  # or after this much time on one solver and problem
const DONE = 1e-10                       # or once the relative suboptimality is below this
const K1, K2 = QUICK ? (5, 25) : (20, 220)
TIME_LIMIT[] = QUICK ? 5.0 : 60.0

problems() = QUICK ?
  [glasso(30), iht_ls(n = 100, p = 300, s = 5), iht_dtrace(p = 12, n = 100, s = 4),
   ls_mcp(n = 100, p = 300, s = 5), quantile_l1(n = 100, p = 20, s = 3), tv_denoise(n = 200)] :
  [glasso(50), glasso(200), iht_ls(), iht_dtrace(), ls_mcp(), quantile_l1(),
   quantile_l1(n = 2000, p = 200, s = 20), tv_denoise(n = 1000), tv_denoise(n = 10_000)]

# the fastest of a few runs; a run slower than RUN_CAP is not repeated
function timed(f; budget = 0.3, minreps = 3, maxreps = 5)
  t = @elapsed out = f()
  times = [t]
  while t <= RUN_CAP && (length(times) < minreps || (sum(times) < budget && length(times) < maxreps))
    push!(times, @elapsed f())
  end
  out, minimum(times)
end

errstring(err) = replace(first(sprint(showerror, err), 160), ',' => ';', '\n' => ' ')
relsub(F, Fstar) = isfinite(F) ? max((F - Fstar) / max(abs(Fstar), 1.0), 0.0) : Inf

bytes_after(f, k) = (f(k); @allocated f(k))
elapsed_min(f, k) = minimum(@elapsed(f(k)) for _ in 1:3)

function sweep(io, P, R, Fstar)
  spent, best, lastF, flat = 0.0, Inf, NaN, 0
  for tol in TOLS
    local x, it, t
    try
      (x, it), t = timed(() -> R.run(tol))
    catch err
      @printf "    %-22s tol = %.0e failed: %s\n" R.name tol errstring(err)
      @printf(io, "%s,%s,%s,%s,%g,NaN,-1,NaN,%.12g,%s\n", P.family, P.name, R.name, R.package, tol, Fstar, errstring(err))
      break
    end
    F = P.objective(x)
    @printf(io, "%s,%s,%s,%s,%g,%.6g,%d,%.12g,%.12g,ok\n", P.family, P.name, R.name, R.package, tol, t, it, F, Fstar)
    flush(io)
    spent += t
    best = min(best, P.convex ? relsub(F, Fstar) : F)
    flat = isfinite(F) && isfinite(lastF) && abs(F - lastF) <= 1e-13 * max(abs(F), 1.0) ? flat + 1 : 0
    lastF = F
    # a run that ends at NaN or Inf (off the domain) does not recover at tighter tolerances
    (!isfinite(F) || t > RUN_CAP || spent > SOLVER_CAP || (P.convex && best < DONE) || flat >= 2) && break
  end
  @printf "    %-22s %s %.3g, sweep %.1f s\n" R.name (P.convex ? "best rel. suboptimality" : "best F") best spent
end

function per_iteration(io, P, R)
  R.iters === nothing && return
  try
    R.iters(2)
    b = (bytes_after(R.iters, K2) - bytes_after(R.iters, K1)) / (K2 - K1)
    t = (elapsed_min(R.iters, K2) - elapsed_min(R.iters, K1)) / (K2 - K1)
    @printf(io, "%s,%s,%s,%s,%.1f,%.6g,ok\n", P.family, P.name, R.name, R.package, b, t)
  catch err
    @printf(io, "%s,%s,%s,%s,NaN,NaN,%s\n", P.family, P.name, R.name, R.package, errstring(err))
  end
  flush(io)
end

# The graphical lasso from a first step that leaves the cone: ProximalAlgorithms'
# solvers with an initial step γ₀ (they backtrack from it), and ProximalSolvers'
# gradient solvers, whose first trial step is 1 and leaves the cone as well.
function inf_probe(io, p; γ0s = (1.0, 10.0))
  rng = MersenneTwister(7)
  Ω0 = Matrix(SymTridiagonal(fill(1.0, p), fill(0.45, p - 1)))
  A = randn(rng, 2p, p) / cholesky(Ω0).U'
  S = Matrix(Symmetric(A'A / 2p))
  W = fill(0.1, p, p); W[diagind(W)] .= 0
  f, g = LogDetLoss(S), NormL1(1.0, W)
  F(X) = f(X) + g(X)
  X0 = Matrix(1.0I, p, p)
  Fstar = F(first(admm(f, g, X0, 1e-12)))
  report(name, package, γ0, run) = try
    (X, it), t = timed(run; minreps = 1, maxreps = 1)
    @printf(io, "%d,%s,%s,%s,%.6g,%d,%.12g,%.12g,ok\n", p, name, package, γ0, t, it, F(X), Fstar)
  catch err
    @printf(io, "%d,%s,%s,%s,NaN,-1,NaN,%.12g,%s\n", p, name, package, γ0, Fstar, errstring(err))
  end
  for (name, m) in GRAD_METHODS
    report(name, "ProximalSolvers", 1.0, () -> ours(m, f, g, X0, 1e-8))
  end
  for γ0 in γ0s, (name, alg, _) in FB_ALGS
    report(name, "ProximalAlgorithms", γ0, () -> run_pa(alg, 1e-8; x0 = X0, f = PASmooth(f), g = PAProx(g),
                                                       gamma = γ0, adaptive = true, maxit = 100_000))
  end
  flush(io)
end

function main()
  println("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", BLAS threads = ", BLAS.get_num_threads(),
          QUICK ? " (quick run)" : "")
  mkpath(OUT)
  sw = open(joinpath(OUT, "sweep.csv"), "w")
  println(sw, "family,problem,solver,package,tol,time,iterations,F,Fstar,status")
  al = open(joinpath(OUT, "per_iteration.csv"), "w")
  println(al, "family,problem,solver,package,bytes,time,status")
  pr = open(joinpath(OUT, "inf_probe.csv"), "w")
  println(pr, "p,solver,package,gamma0,time,iterations,F,Fstar,status")

  # compile every path on a small instance of each problem first
  QUICK || foreach(P -> foreach(R -> try R.run(1e-4) catch end, P.runs),
                   [glasso(10), iht_ls(n = 40, p = 80, s = 3), iht_dtrace(p = 8, n = 60, s = 2),
                    ls_mcp(n = 40, p = 80, s = 3), quantile_l1(n = 40, p = 10, s = 2), tv_denoise(n = 100)])

  for P in problems()
    @printf "\n### %s\n" P.name
    Fstar = NaN
    if P.reference !== nothing
      tref = @elapsed (Fstar, how) = P.reference()
      @printf "  F* = %.12g (%s, %.1f s)\n" Fstar how tref
    end
    for R in P.runs
      sweep(sw, P, R, Fstar)
      per_iteration(al, P, R)
    end
    flush(stdout)
  end

  println("\n### graphical lasso from an infeasible first step")
  for (p, γ0s) in (QUICK ? ((20, (1.0, 10.0)),) : ((50, (1.0, 10.0)), (200, (1.0,))))
    inf_probe(pr, p; γ0s)
  end
  close(sw); close(al); close(pr)
end

main()
