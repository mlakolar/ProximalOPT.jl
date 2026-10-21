using Test


tests = [
  "prox_solvers",
  "linesearch",
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
