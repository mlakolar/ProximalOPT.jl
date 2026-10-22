# Markdown tables from the results of lasso_bench.jl.
#
#     julia --project=benchmark benchmark/lasso_summary.jl          # tables
#     julia --project=benchmark benchmark/lasso_summary.jl --raw    # and every run

using Printf

const RESULTS = joinpath(@__DIR__, "results")
const TARGETS = (1e-6, 1e-10)

# rows of a CSV file written by lasso_bench.jl (no field contains a comma)
function read_csv(file)
  lines = collect(eachline(joinpath(RESULTS, file)))
  header = Symbol.(split(lines[1], ","))
  [NamedTuple{Tuple(header)}(Tuple(_parse.(split(l, ",")))) for l in lines[2:end]]
end

_parse(s) = (x = tryparse(Float64, s); x === nothing ? String(s) : x)

# time of the cheapest run that reached the target, or nothing
function time_to(rows, target, key)
  ok = filter(r -> getproperty(r, key) <= target, rows)
  isempty(ok) ? nothing : minimum(r -> r.time, ok)
end

fmt_ms(t) = t < 0.0995 ? @sprintf("%.1f", 1000t) : @sprintf("%.0f", 1000t)
fmt_s(t) = t < 0.995 ? @sprintf("%.2f", t) : @sprintf("%.1f", t)

function table(solvers, cols, colname, cell)
  println("| solver | ", join(colname.(cols), " | "), " |")
  println("|---|", repeat("---:|", length(cols)))
  for s in solvers
    println("| ", s, " | ", join((cell(s, c) for c in cols), " | "), " |")
  end
end

function main()
  S = read_csv("single.csv")
  sweeps = filter(r -> r.package != "defaults", S)
  defaults = filter(r -> r.package == "defaults", S)
  cases = unique([(r.scenario, r.lambda_frac) for r in sweeps])
  solvers = unique([r.solver for r in sweeps])
  descs = Dict(r.scenario => r.desc for r in S)

  println("Designs: ", join(["$k: $(descs[k])" for k in sort(collect(keys(descs)))], "; "), ".\n")

  for target in TARGETS
    @printf "## One λ, from β = 0: time (ms) to relative suboptimality %.0e\n\n" target
    cell(s, c) = begin
      rs = filter(r -> r.solver == s && (r.scenario, r.lambda_frac) == c, sweeps)
      isempty(rs) && return "–"
      t = time_to(rs, target, :rel_subopt)
      t === nothing ? @sprintf("✗ (%.0e)", minimum(r -> r.rel_subopt, rs)) : fmt_ms(t)
    end
    table(solvers, cases, c -> @sprintf("%s, %.2f λmax", c[1], c[2]), cell)
    nnz = [first(r.nnz_star for r in sweeps if (r.scenario, r.lambda_frac) == c) for c in cases]
    gram = [maximum(r.gram_time for r in sweeps if (r.scenario, r.lambda_frac) == c) for c in cases]
    println("| nonzeros in solution | ", join(Int.(nnz), " | "), " |")
    println("| forming X'X/n (ms) | ", join((g > 0 ? fmt_ms(g) : "–" for g in gram), " | "), " |\n")
  end
  println("The X'X/n rows do not include forming X'X/n. ✗: target not reached; best value in parentheses.\n")

  println("## Package defaults\n")
  for c in cases
    @printf "### %s, λ = %.2f λmax (%d nonzeros)\n\n" c[1] c[2] first(r.nnz_star for r in sweeps if (r.scenario, r.lambda_frac) == c)
    println("| solver | time | rel. suboptimality | max abs. error | support errors | KKT residual |")
    println("|---|---:|---:|---:|---:|---:|")
    for r in filter(r -> (r.scenario, r.lambda_frac) == c, defaults)
      if isnan(r.rel_subopt)
        println("| ", r.solver, " | error | | | | |")
      else
        @printf "| %s | %s ms | %.1e | %.1e | %d | %.1e |\n" r.solver fmt_ms(r.time) r.rel_subopt r.max_abs_diff Int(r.support_errors) r.kkt
      end
    end
    println()
  end

  P = read_csv("path.csv")
  pscen = unique([r.scenario for r in P])
  psolvers = unique([r.solver for r in P])
  nλ = Int(maximum(r.n_lambda for r in P))
  @printf "## Path of %d λ values from λmax to λmax/100, warm started: seconds until every λ is within the target\n\n" nλ
  cols = [(sc, t) for t in TARGETS for sc in pscen]
  pcell(s, c) = begin
    rs = filter(r -> r.solver == s && r.scenario == c[1], P)
    isempty(rs) && return "–"
    t = time_to(rs, c[2], :max_rel_subopt)
    t === nothing ? "✗" : fmt_s(t + rs[1].gram_time)
  end
  table(psolvers, cols, c -> @sprintf("%s, %.0e", c...), pcell)
  println("\nThe X'X/n rows include forming X'X/n. ✗: target not reached within the sweep.")

  if "--raw" in ARGS
    println("\n## Every run (tolerance: time / relative suboptimality)\n")
    for c in cases, s in solvers
      rs = filter(r -> r.solver == s && (r.scenario, r.lambda_frac) == c, sweeps)
      isempty(rs) && continue
      @printf "%s %.2f %-24s" c[1] c[2] s
      foreach(r -> @printf(" %.0e: %s ms / %.0e", r.tol, fmt_ms(r.time), r.rel_subopt), rs)
      println()
    end
  end
end

main()
