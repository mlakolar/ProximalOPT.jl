using FactCheck

using ProximalOPT

tests = [
	"diff_fun",
  "prox_solvers",
  "prox_fun",
  "lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end

FactCheck.exitstatus()


