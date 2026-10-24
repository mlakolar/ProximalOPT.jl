using Test


tests = [
  "prox_solvers",
  "linesearch",
  "call_limits",
  "admm",
  "active_set",
  "group_lasso",
  "low_rank",
  "coordinate_descent",
  "group_penalties",
  "nonconvex_penalties",
  "glm",
  "aqua",
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
    t = @elapsed include(f)
    println("done (took $t seconds).")
end
