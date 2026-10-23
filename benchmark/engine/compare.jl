# Compare runs of old-cd/run.jl and old/run.jl with runs of new/run.jl, case by
# case:
#
#     julia benchmark/engine/compare.jl first.csv second.csv [--all]
#     julia benchmark/engine/compare.jl old1.csv,old2.csv new1.csv,new2.csv
#
# Each side is one run or several, separated by commas; a case's time is the
# minimum over the runs of its side, since load on the machine only ever adds
# time (ab.sh makes interleaved runs). For each case in both sides it prints the
# times, their ratio (second over first), the passes, the largest difference
# between the solutions and the relative difference of the objectives (from the
# first run of each side); then the geometric mean of the ratios by group and
# the cases more than 10% slower. Only standard libraries, so any environment
# runs it.

using Printf, Serialization, Statistics

struct Row
  key::String
  group::String
  time::Float64
  passes::Int
  converged::Bool
  objective::Float64
  nnz::Int
end

function read_rows(csv)
  rows = Dict{String, Row}()
  order = String[]
  for line in eachline(csv)
    (startswith(line, "#") || startswith(line, "group,problem,solver,")) && continue
    f = split(line, ',')
    # group, problem, solver, then 8 numbers; an old problem name may hold a comma
    group, solver = f[1], f[end - 8]
    problem = join(f[2:end - 9], ',')
    key = string(group, " | ", problem, " | ", solver)
    rows[key] = Row(key, group, parse(Float64, f[end - 7]), parse(Int, f[end - 3]),
                    parse(Bool, f[end - 2]), parse(Float64, f[end - 1]), parse(Int, f[end]))
    push!(order, key)
  end
  rows, order
end

# the rows of several runs, each case with its minimum time over the runs and
# the rest from its first run; the runs may hold different cases, as those of
# old-cd/run.jl and old/run.jl
function read_side(spec)
  files = split(spec, ',')
  rows, order = Dict{String, Row}(), String[]
  solutions = Dict{String, Vector{Float64}}()
  for f in files
    more, morder = read_rows(f)
    sols = deserialize(replace(String(f), r"\.csv$" => ".jls"))
    for key in morder
      r = more[key]
      if !haskey(rows, key)
        rows[key] = r
        push!(order, key)
        haskey(sols, key) && (solutions[key] = sols[key])
      elseif r.time < rows[key].time
        rows[key] = Row(key, r.group, r.time, rows[key].passes, rows[key].converged, rows[key].objective, rows[key].nnz)
      end
    end
  end
  rows, order, solutions, files
end

function main(args)
  specs = filter(a -> !startswith(a, "--"), args)
  length(specs) == 2 || error("usage: compare.jl first.csv[,more.csv] second.csv[,more.csv] [--all]")
  showall = "--all" in args
  r1, order, s1, f1 = read_side(specs[1])
  r2, _, s2, f2 = read_side(specs[2])
  println("first:  ", join(f1, ", "), "\n        ", readline(f1[1]))
  println("second: ", join(f2, ", "), "\n        ", readline(f2[1]))

  ratios = Dict{String, Vector{Float64}}()
  slow = String[]
  @printf "\n%-70s %10s %10s %6s %13s %9s %9s\n" "case" "first ms" "second ms" "ratio" "passes" "max|Δx|" "rel ΔF"
  for key in order
    haskey(r2, key) || continue
    a, b = r1[key], r2[key]
    ratio = b.time / a.time
    push!(get!(ratios, a.group, Float64[]), ratio)
    x1, x2 = get(s1, key, nothing), get(s2, key, nothing)
    dx = (x1 === nothing || x2 === nothing || length(x1) != length(x2)) ? NaN : maximum(abs, x1 - x2; init = 0.0)
    dF = isnan(a.objective) ? NaN : abs(b.objective - a.objective) / max(abs(a.objective), eps())
    ratio > 1.1 && push!(slow, @sprintf("%s: %.2f×", key, ratio))
    @printf "%-70s %10.3f %10.3f %6.2f %6d/%-6d %9.1e %9.1e%s\n" first(key, 70) 1e3a.time 1e3b.time ratio a.passes b.passes dx dF (b.converged ? "" : "  second not converged")
  end
  println("\ngeometric mean of the time ratios (second / first):")
  all = Float64[]
  for (g, v) in sort(collect(ratios); by = first)
    append!(all, v)
    @printf "  %-16s %5.2f   (%d cases, range %.2f–%.2f)\n" g exp(mean(log.(v))) length(v) minimum(v) maximum(v)
  end
  isempty(all) || @printf "  %-16s %5.2f   (%d cases)\n" "all" exp(mean(log.(all))) length(all)
  if !isempty(slow)
    println("\nmore than 10% slower:")
    foreach(s -> println("  ", s), slow)
  end
  missing2 = setdiff(order, keys(r2))
  (showall && !isempty(missing2)) && (println("\nonly in the first run:"); foreach(k -> println("  ", k), missing2))
end

main(ARGS)
