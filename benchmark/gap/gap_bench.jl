# Duality-gap stopping and gap-safe screening against today's rules, on the
# lasso designs A–E: one λ (0.1 and 0.02 λmax) from zero, and a path of 50
# levels from λmax to λmax/100. For each run: the time (fastest of three), the
# passes or iterations, and the relative duality gap (P - D)/|P| reached,
# computed afterwards by duality_gap.
#
#     julia --project=benchmark/gap benchmark/gap/gap_bench.jl [--quick] [--out FILE]

include(joinpath(@__DIR__, "problems.jl"))

const QUICK = "--quick" in ARGS
const OUT = let i = findfirst(==("--out"), ARGS); i === nothing ? joinpath(@__DIR__, "gap_results.csv") : ARGS[i+1] end

relgap(f, g, x) = (d = duality_gap(f, g, x); max(d.gap, 0.0) / abs(d.primal))

const CD = [
  ("CD default", CDSolver()),
  ("CD gaptol 1e-6", CDSolver(; gaptol = 1e-6)),
  ("CD gaptol 1e-6 + gapsafe", CDSolver(; gaptol = 1e-6, gapsafe = true)),
  ("CD default + gapsafe", CDSolver(; gapsafe = true)),
]

function single(io, d, f, λmax, p)
  for frac in (0.1, 0.02)
    g = NormL1(frac * λmax)
    for (label, s) in CD
      t, r = fastest(() -> solve!(SparseIterate(p), f, g, s))
      gap = relgap(f, g, r.minimizer)
      @printf "%s  %-5.2f  %-28s %9.4f s  %7d passes  gap %.1e\n" d.name frac label t r.passes gap
      println(io, join((d.name, "single", frac, label, t, r.passes, gap), ","))
    end
    for (label, opt) in (("APG default", ProximalOptions(; maxiter = 100_000)),
                         ("APG gaptol 1e-6", ProximalOptions(; maxiter = 100_000, gaptol = 1e-6)))
      t, r = fastest(() -> solve!(zeros(p), f, g, AccProxGradDescent(), opt))
      gap = relgap(f, g, r.minimizer)
      @printf "%s  %-5.2f  %-28s %9.4f s  %7d iters   gap %.1e\n" d.name frac label t r.iterations gap
      println(io, join((d.name, "single", frac, label, t, r.iterations, gap), ","))
    end
  end
end

function path(io, d, f, λmax, p)
  λs = λmax .* 10 .^ range(0, -2; length = 50)
  for (label, s) in CD
    t, r = fastest(() -> solve_path(f, NormL1(1.0), λs, s))
    gap = maximum(relgap(f, NormL1(λ), SparseIterate(Vector(x))) for (λ, x) in zip(r.λ, r.solutions))
    @printf "%s  path   %-28s %9.4f s  %7d passes  max gap %.1e\n" d.name label t sum(r.passes) gap
    println(io, join((d.name, "path", "", label, t, sum(r.passes), gap), ","))
  end
end

function main()
  f, λmax, p = design((name = "warmup", n = 50, p = 100, s = 5, ρ = 0.0, common = 0.0))
  devnull_io = IOBuffer()
  redirect_stdout(devnull) do
    single(devnull_io, (name = "w",), f, λmax, p); path(devnull_io, (name = "w",), f, λmax, p)
  end
  designs = QUICK ? [(name = "A", n = 200, p = 1000, s = 20, ρ = 0.0, common = 0.0)] : DESIGNS
  open(OUT, "w") do io
    println(io, "design,kind,lambda,solver,seconds,passes,relgap")
    for d in designs
      f, λmax, p = design(d)
      single(io, d, f, λmax, p)
      path(io, d, f, λmax, p)
      flush(io)
    end
  end
  println("wrote ", OUT)
end

main()
