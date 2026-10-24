using Test


tests = [
  "prox_solvers",
  "nonconvex",
  "linesearch",
  "image_models",
  "call_limits",
  "admm",
  "product_space",
  "active_set",
  "group_lasso",
  "low_rank",
  "coordinate_descent",
  "sparse_x",
  "group_penalties",
  "glm",
  "duality_gap",
  "quic",
  "examples",
  "aqua",
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
    t = @elapsed include(f)
    println("done (took $t seconds).")
end
