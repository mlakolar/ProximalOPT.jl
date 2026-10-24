using Test


tests = [
  "prox_solvers",
  "linesearch",
  "image_models",
  "call_limits",
  "admm",
  "active_set",
  "group_lasso",
  "low_rank",
  "coordinate_descent",
  "sparse_x",
  "glm",
  "examples",
  "aqua",
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
    t = @elapsed include(f)
    println("done (took $t seconds).")
end
