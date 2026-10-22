# Timing and recording for old/run.jl and new/run.jl. A run records, for each
# case, the minimum and median time, the bytes allocated by one call, the
# passes, whether the solver converged, the objective, the number of nonzeros,
# and the solution, so that compare.jl can check that the old and the new code
# compute the same thing.

using Printf, Serialization, Dates, InteractiveUtils

include(joinpath(@__DIR__, "problems.jl"))

"""
Run `f()` at least `minreps` times, and more until the total time reaches
`budget` seconds (at most `maxreps` runs). Returns the output of the first run,
the minimum and the median time, and the number of runs: other load on the
machine only ever adds time, so the minimum is the best estimate.
"""
function timed(f; budget = 1.0, minreps = 3, maxreps = 15)
  GC.gc()
  t = @elapsed out = f()
  times = [t]
  while length(times) < minreps || (sum(times) < budget && length(times) < maxreps)
    push!(times, @elapsed f())
  end
  out, minimum(times), median(times), length(times)
end

struct Case
  group::String
  problem::String
  solver::String
end

key(c::Case) = string(c.group, " | ", c.problem, " | ", c.solver)

mutable struct Recorder
  label::String
  rows::Vector{NamedTuple}
  solutions::Dict{String, Vector{Float64}}
  only::Union{Nothing, Regex}
end

Recorder(label; only = nothing) = Recorder(label, NamedTuple[], Dict{String, Vector{Float64}}(), only)

"""
    run!(f, rec, case; objective, budget, warmup = true)

Time `f()`, which returns `(x, passes, converged)` with `x` a dense vector of
the solution (a matrix solution as `vec`); call it with a `do` block.
`objective(x)` gives the value recorded for the solution. The first call
compiles, unless `warmup` is false.
"""
function run!(f, rec::Recorder, case::Case; objective = x -> NaN, budget = 1.0,
              minreps = 3, maxreps = 15, warmup = true)
  rec.only === nothing || occursin(rec.only, key(case)) || return nothing
  local x, passes, converged, tmin, tmed, reps, bytes
  try
    warmup && f()
    bytes = @allocated f()
    (x, passes, converged), tmin, tmed, reps = timed(f; budget, minreps, maxreps)
  catch err
    @printf "  %-58s FAILED: %s\n" key(case) first(sprint(showerror, err), 200)
    return nothing
  end
  x = Vector{Float64}(x)
  obj = objective(x)
  push!(rec.rows, (group = case.group, problem = case.problem, solver = case.solver,
                   time_min = tmin, time_med = tmed, reps = reps, bytes = bytes,
                   passes = passes, converged = converged, objective = obj,
                   nnz = count(!iszero, x)))
  rec.solutions[key(case)] = x
  @printf "  %-58s %10.3f ms  %6d passes  nnz %6d  F %.10e%s\n" key(case) 1e3tmin passes count(!iszero, x) obj (converged ? "" : "  NOT CONVERGED")
  flush(stdout)
  nothing
end

function environment_summary()
  string("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", ", Sys.CPU_THREADS, " threads, BLAS threads = ",
         LinearAlgebra.BLAS.get_num_threads(), ", ", gethostname())
end

"""
Write the rows as CSV and the solutions with Serialization, to
`results/<label>_<timestamp>.csv` and `.jls`. Returns the CSV path.
"""
function save(rec::Recorder; dir = joinpath(@__DIR__, "results"))
  mkpath(dir)
  stamp = Dates.format(now(), "yyyymmddTHHMMSS")
  base = joinpath(dir, string(rec.label, "_", stamp))
  open(base * ".csv", "w") do io
    println(io, "# ", environment_summary())
    println(io, "group,problem,solver,time_min,time_med,reps,bytes,passes,converged,objective,nnz")
    for r in rec.rows
      @printf(io, "%s,%s,%s,%.6g,%.6g,%d,%d,%d,%s,%.15g,%d\n", _csv(r.group), _csv(r.problem), _csv(r.solver),
              r.time_min, r.time_med, r.reps, r.bytes, r.passes, r.converged, r.objective, r.nnz)
    end
  end
  serialize(base * ".jls", rec.solutions)
  base * ".csv"
end

_csv(s) = replace(s, ',' => ';')

# command line: `--only REGEX` restricts the cases to those whose key matches
function only_from_args(args)
  i = findfirst(==("--only"), args)
  i === nothing ? nothing : Regex(args[i + 1])
end
