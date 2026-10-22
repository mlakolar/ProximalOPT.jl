#!/bin/sh
# Interleaved runs of the old and the new code, then the comparison: load on a
# shared machine changes over minutes, so the two sides alternate and each case
# keeps its fastest run.
#
#     benchmark/engine/ab.sh [ROUNDS] [REGEX]        # from the root of the repository
set -e
rounds=${1:-2}
only=${2:-.}
dir=$(dirname "$0")
old=""; new=""
for i in $(seq "$rounds"); do
  o=$(julia --project="$dir/old" "$dir/old/run.jl" --only "$only" | sed -n 's/^wrote //p')
  n=$(julia --project="$dir/new" "$dir/new/run.jl" --only "$only" | sed -n 's/^wrote //p')
  old="${old:+$old,}$o"; new="${new:+$new,}$n"
done
julia "$dir/compare.jl" "$old" "$new"
