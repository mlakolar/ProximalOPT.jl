# Lasso benchmark: ProximalOPT's proximal gradient solvers against
# CoordinateDescent.jl, with glmnet (GLMNet.jl) and Lasso.jl for reference.
#
#     julia --project=benchmark benchmark/lasso_bench.jl            # about an hour
#     julia --project=benchmark benchmark/lasso_bench.jl --quick    # a minute
#
# 1. One λ, starting from β = 0: each solver is run over a sweep of its
#    tolerance, and each run records its time and accuracy. This gives the time
#    a solver needs to reach a given relative suboptimality (F - F*) / F*.
# 2. Each solver at its package defaults.
# 3. A path of λ values from λmax to λmax/100, each warm started from the
#    previous solution.
#
# Results are written to benchmark/results/single.csv and path.csv; print
# tables from them with lasso_summary.jl. See README.md.

include(joinpath(@__DIR__, "lasso_setup.jl"))
BLAS.set_num_threads(1)

const QUICK = "--quick" in ARGS
const OUT = joinpath(@__DIR__, "results")

const SCENARIOS = QUICK ?
  [(name = "A", desc = "n=300 p=1000 independent", n = 300, p = 1000, s = 20, ρ = 0.0, common = 0.0, gram = true)] :
  [(name = "A", desc = "n=500 p=2500 independent",      n = 500,  p = 2500,  s = 50, ρ = 0.0, common = 0.0, gram = true),
   (name = "B", desc = "n=500 p=2500 AR(1) 0.8",        n = 500,  p = 2500,  s = 50, ρ = 0.8, common = 0.0, gram = true),
   (name = "C", desc = "n=5000 p=500 AR(1) 0.5",        n = 5000, p = 500,   s = 50, ρ = 0.5, common = 0.0, gram = true),
   (name = "D", desc = "n=1000 p=20000 independent",    n = 1000, p = 20000, s = 20, ρ = 0.0, common = 0.0, gram = false),
   (name = "E", desc = "n=500 p=2500 equicorrelated 0.5", n = 500, p = 2500, s = 50, ρ = 0.0, common = 0.5, gram = true)]
const LAMBDA_FRACS = QUICK ? (0.1,) : (0.1, 0.02)
const N_LAMBDA = QUICK ? 10 : 50
const TOLS = QUICK ? 10.0 .^ (-2:-2:-14) : 10.0 .^ (-1:-1:-16)
const PATH_TOLS = QUICK ? 10.0 .^ (-4:-2:-14) : 10.0 .^ (-3:-1:-14)
const RUN_CAP = 20.0       # stop a sweep after a run this slow (seconds)
const SOLVER_CAP = 90.0    # or after this much time on one solver in one case
const DONE = 1e-11         # or once the relative suboptimality is below this

function warmup()
  P = make_problem("warmup", "warmup"; n = 50, p = 100, s = 5)
  λ = 0.1 * P.λmax
  λs = P.λmax .* 10 .^ range(0, -2; length = 5)
  foreach(S -> S.run(P, λ, 1e-6), SOLVERS)
  foreach(d -> d[2](P, λ), DEFAULT_RUNS)
  foreach(S -> S.run(P, λs, 1e-6), PATH_SOLVERS)
  nothing
end

function main()
  println(env_summary(), QUICK ? " (quick run)" : "")
  warmup()
  mkpath(OUT)

  single = open(joinpath(OUT, "single.csv"), "w")
  println(single, "scenario,desc,lambda_frac,nnz_star,solver,package,tol,time,gram_time,rel_subopt,max_abs_diff,support_errors,nnz,kkt")
  paths = open(joinpath(OUT, "path.csv"), "w")
  println(paths, "scenario,desc,solver,package,tol,time,gram_time,max_rel_subopt,max_abs_diff,max_support_errors,n_lambda")

  for sc in SCENARIOS
    P = make_problem(sc.name, sc.desc; n = sc.n, p = sc.p, s = sc.s, ρ = sc.ρ, common = sc.common, gram = sc.gram)
    @printf "\n### %s: %s (X'X/n %.2f s)\n" sc.name sc.desc P.gram_time
    flush(stdout)

    for frac in LAMBDA_FRACS
      λ = frac * P.λmax
      tref = @elapsed (βstar, Fstar, kkt) = reference(P, λ)
      nnzstar = count(!iszero, βstar)
      @printf "  λ = %.2f λmax: nnz(β*) = %d, KKT residual of the reference %.1e (%.1f s)\n" frac nnzstar kkt tref

      row(solver, package, tol, gram_time, β, t) = @printf(single, "%s,%s,%g,%d,%s,%s,%g,%.6g,%.6g,%.6g,%.6g,%d,%d,%.6g\n",
        P.name, P.desc, frac, nnzstar, solver, package, tol, t, gram_time,
        relsubopt(objective(P, β, λ), Fstar), maximum(abs, β - βstar), support_errors(β, βstar),
        count(!iszero, β), kkt_residual(P, β, λ))

      for S in SOLVERS
        (S.needs_gram && P.G === nothing) && continue
        spent, best = 0.0, Inf
        for tol in TOLS
          local β, t
          try
            β, t = timed(() -> S.run(P, λ, tol); budget = 0.3)
          catch err
            @printf "    %-24s tol = %.0e failed: %s\n" S.name tol first(sprint(showerror, err), 120)
            break
          end
          row(S.name, S.package, tol, S.needs_gram ? P.gram_time : 0.0, β, t)
          spent += t
          best = min(best, relsubopt(objective(P, β, λ), Fstar))
          (t > RUN_CAP || spent > SOLVER_CAP || best < DONE) && break
        end
        @printf "    %-24s best rel. suboptimality %.1e, sweep %.1f s\n" S.name best spent
        flush(single); flush(stdout)
      end

      for (name, run) in DEFAULT_RUNS
        try
          β, t = timed(() -> run(P, λ); budget = 0.3)
          row(name, "defaults", NaN, 0.0, β, t)
        catch err
          # recorded with NaN accuracy, reported as an error
          @printf "    defaults %-15s failed: %s\n" name first(sprint(showerror, err), 120)
          @printf(single, "%s,%s,%g,%d,%s,%s,%g,%.6g,%.6g,%.6g,%.6g,%d,%d,%.6g\n", P.name, P.desc, frac,
                  nnzstar, name, "defaults", NaN, NaN, 0.0, NaN, NaN, -1, -1, NaN)
        end
      end
      flush(single)
    end

    λs = P.λmax .* 10 .^ range(0, -2; length = N_LAMBDA)
    tref = @elapsed ref = path_reference(P, λs)
    Fref = [objective(P, ref[k], λs[k]) for k in eachindex(λs)]
    @printf "  path: %d λ from λmax to λmax/100, nnz at the end %d (reference %.1f s)\n" N_LAMBDA count(!iszero, ref[end]) tref
    for S in PATH_SOLVERS
      (S.needs_gram && P.G === nothing) && continue
      spent, best = 0.0, Inf
      for tol in PATH_TOLS
        local B, t
        try
          B, t = timed(() -> S.run(P, λs, tol); budget = 0.3, maxreps = 3)
        catch err
          @printf "    path %-19s tol = %.0e failed: %s\n" S.name tol first(sprint(showerror, err), 120)
          break
        end
        K = length(B)
        sub = maximum(relsubopt(objective(P, B[k], λs[k]), Fref[k]) for k in 1:K)
        dif = maximum(maximum(abs, B[k] - ref[k]) for k in 1:K)
        sup = maximum(support_errors(B[k], ref[k]) for k in 1:K)
        @printf(paths, "%s,%s,%s,%s,%g,%.6g,%.6g,%.6g,%.6g,%d,%d\n", P.name, P.desc, S.name, S.package, tol, t,
                S.needs_gram ? P.gram_time : 0.0, K < length(λs) ? Inf : sub, dif, sup, K)
        spent += t
        best = min(best, sub)
        (t > RUN_CAP || spent > SOLVER_CAP || best < DONE) && break
      end
      @printf "    path %-19s best max rel. suboptimality %.1e, sweep %.1f s\n" S.name best spent
      flush(paths); flush(stdout)
    end
  end
  close(single)
  close(paths)
end

main()
