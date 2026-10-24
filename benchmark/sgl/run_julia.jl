# ProximalSolvers' sparse-group-lasso path on a problem of problems.jl, at one
# mixture α, on the grid written there: the time of the whole path (median of
# REPS runs after one run that compiles), and the solutions at every level.
#
#   julia --project=benchmark/sgl benchmark/sgl/run_julia.jl PROBDIR ALPHA OUTDIR [REPS]

using ProximalBase, ProximalSolvers, Random, LinearAlgebra, Printf

include(joinpath(@__DIR__, "common.jl"))

function main()
  d, α, out = ARGS[1], parse(Float64, ARGS[2]), ARGS[3]
  reps = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3
  X, y, groups = readproblem(d)
  n, p = size(X)
  λs = Vector(reinterpret(Float64, read(joinpath(d, "lambda_$(α).bin"))))
  f = (1 / n) * LeastSquaresLoss(y, X)
  g = Scaled(SparseGroupNormL2(α, 1 - α, groups), 1.0)
  for (label, method) in (("ProximalSolvers", CDSolver()), ("ProximalSolvers-gapsafe", CDSolver(; gapsafe = true)))
    fit() = (Random.seed!(1); solve_path(f, g, λs, method; x = AtomIterate(p, groups)))
    path = fit()                          # compiles
    times = Float64[]
    for _ in 1:reps
      GC.gc()
      t = @elapsed path = fit()
      push!(times, t)
    end
    B = reduce(hcat, Vector.(path.solutions))
    mkpath(out)
    write(joinpath(out, "beta_$(label).bin"), B)
    open(joinpath(out, "time_$(label).txt"), "w") do io
      println(io, join(times, " "))
      println(io, "passes=", sum(path.passes), " checks=", sum(path.kkt_checks), " converged=", all(path.converged))
    end
    @printf "%s α=%.2f %s: median %.3f s over %d runs, %d passes\n" basename(d) α label sort(times)[(reps + 1) ÷ 2] reps sum(path.passes)
  end
end

main()
