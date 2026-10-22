# Compare two runs of old/run.jl or new/run.jl, case by case:
#
#     julia benchmark/engine/compare.jl results/old_<stamp>.csv results/new_<stamp>.csv [--all]
#
# For each case in both runs it prints the minimum times, their ratio (second
# run over first), the passes, the largest difference between the solutions and
# the relative difference of the objectives; then the geometric mean of the
# ratios by group and the cases more than 10% slower. Only standard libraries,
# so any environment runs it.

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
    (startswith(line, "#") || startswith(line, "group,")) && continue
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

function main(args)
  files = filter(a -> !startswith(a, "--"), args)
  length(files) == 2 || error("usage: compare.jl first.csv second.csv [--all]")
  showall = "--all" in args
  r1, order = read_rows(files[1])
  r2, _ = read_rows(files[2])
  s1 = deserialize(replace(files[1], r"\.csv$" => ".jls"))
  s2 = deserialize(replace(files[2], r"\.csv$" => ".jls"))
  println("first:  ", files[1], "\n        ", readline(files[1]))
  println("second: ", files[2], "\n        ", readline(files[2]))

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
