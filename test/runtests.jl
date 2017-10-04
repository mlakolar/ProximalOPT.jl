using FactCheck

using ProximalBase
using ProximalOPT

tests = [
  "prox_solvers",
  "lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()
