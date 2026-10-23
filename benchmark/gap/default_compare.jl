# Compares the CSV files of default_ab.jl from two checkouts: for each case the
# time ratio new/old of the fastest runs and of the median runs over the files
# (rounds) of each side, and whether the passes agree.
#
#     julia benchmark/gap/default_compare.jl OLD1,OLD2,... NEW1,NEW2,...

using Printf, Statistics

function load(files)
  d = Dict{Tuple{String, String}, Tuple{Vector{Float64}, Int}}()
  for file in split(files, ","), line in Iterators.drop(eachline(file), 1)
    a, c, t, n = split(line, ",")
    key = (String(a), String(c))
    times, _ = get!(d, key, (Float64[], parse(Int, n)))
    push!(times, parse(Float64, t))
  end
  d
end

old, new = load(ARGS[1]), load(ARGS[2])
rmin, rmed = Float64[], Float64[]
for key in sort(collect(keys(old)))
  haskey(new, key) || continue
  (to, po), (tn, pn) = old[key], new[key]
  push!(rmin, minimum(tn) / minimum(to)); push!(rmed, median(tn) / median(to))
  @printf "%-3s %-24s old %9.4f s  new %9.4f s  fastest %.3f  median %.3f  passes %s\n" key[1] key[2] minimum(to) minimum(tn) rmin[end] rmed[end] (po == pn ? "same ($po)" : "DIFFER $po vs $pn")
end
gm(r) = exp(sum(log, r) / length(r))
@printf "over %d cases: geometric mean of the ratios %.3f (fastest) and %.3f (median); largest %.3f and %.3f\n" length(rmin) gm(rmin) gm(rmed) maximum(rmin) maximum(rmed)
