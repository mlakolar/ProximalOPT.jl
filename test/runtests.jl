using Test


tests = [
  "prox_solvers",
  "linesearch",
  "call_limits",
  "admm",
  "aqua",
  # "lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
    t = @elapsed include(f)
    println("done (took $t seconds).")
end
