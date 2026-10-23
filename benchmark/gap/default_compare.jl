# Compares the CSV files of default_ab.jl from two checkouts: for each case the
# fastest time over the files of each side, the time ratio new/old, and whether
# the passes agree.
#
#     julia benchmark/gap/default_compare.jl OLD1,OLD2,... NEW1,NEW2,...

using Printf

function load(files)
  d = Dict{Tuple{String, String}, Tuple{Float64, Int}}()
  for file in split(files, ","), line in Iterators.drop(eachline(file), 1)
    a, c, t, n = split(line, ",")
    key, tt, pp = (String(a), String(c)), parse(Float64, t), parse(Int, n)
    d[key] = haskey(d, key) ? (min(d[key][1], tt), pp) : (tt, pp)
  end
  d
end

old, new = load(ARGS[1]), load(ARGS[2])
ratios = Float64[]
for key in sort(collect(keys(old)))
  haskey(new, key) || continue
  (to, po), (tn, pn) = old[key], new[key]
  push!(ratios, tn / to)
  @printf "%-3s %-28s old %9.4f s  new %9.4f s  ratio %.3f  passes %s\n" key[1] key[2] to tn tn / to (po == pn ? "same ($po)" : "DIFFER $po vs $pn")
end
@printf "geometric mean ratio %.3f, max %.3f over %d cases\n" exp(sum(log, ratios) / length(ratios)) maximum(ratios) length(ratios)
