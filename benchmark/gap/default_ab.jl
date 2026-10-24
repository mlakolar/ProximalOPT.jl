# The default coordinate descent (no gap) on the lasso designs A–E, to compare
# two checkouts of ProximalSolvers: run it with each, interleaved, and compare
# the CSV files with default_compare.jl. Same passes, and times within noise,
# means that the duality-gap changes cost nothing when they are off.
#
#     julia --project=ENV benchmark/gap/default_ab.jl --out FILE

include(joinpath(@__DIR__, "problems.jl"))

const OUT = ARGS[findfirst(==("--out"), ARGS) + 1]

function cases(io, d, f, λmax, p)
  for frac in (0.1, 0.02), s in (CDSolver(), CDSolver(; order = :cyclic), CDSolver(; order = :cyclic, anderson = 5))
    Random.seed!(1)
    t, r = fastest(() -> (Random.seed!(1); solve!(SparseIterate(p), f, NormL1(frac * λmax), s)))
    println(io, join((d.name, "single $frac $(s.random ? "random" : "cyclic") $(s.anderson)", t, r.passes), ","))
  end
  λs = λmax .* 10 .^ range(0, -2; length = 50)
  t, r = fastest(() -> (Random.seed!(1); solve_path(f, NormL1(1.0), λs, CDSolver())))
  println(io, join((d.name, "path", t, sum(r.passes)), ","))
end

function main()
  f, λmax, p = design((name = "warmup", n = 50, p = 100, s = 5, ρ = 0.0, common = 0.0))
  cases(IOBuffer(), (name = "w",), f, λmax, p)
  open(OUT, "w") do io
    println(io, "design,case,seconds,passes")
    for d in DESIGNS
      f, λmax, p = design(d)
      cases(io, d, f, λmax, p)
      flush(io)
    end
  end
  println("wrote ", OUT)
end

main()
