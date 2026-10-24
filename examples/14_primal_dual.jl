# Generalized lasso with second differences; all buffers and steps can also
# use Float32. Ordinary chain TV has a direct TotalVariation1D prox instead.
using LinearAlgebra, SparseArrays, Random, ProximalBase, ProximalSolvers
rng = MersenneTwister(23)
n = 80
b = sin.(range(0, 2π; length=n)) + .1randn(rng,n)
D = spdiagm(n-2,n,0=>ones(n-2),1=>-2ones(n-2),2=>ones(n-2))
f, g, h = SqrNormL2(1.,b), NormL1(.01), NormL1(.5)
x = zeros(n)
options = ProximalOptions(maxiter=100_000)
res = solve!(x,f,g,h,D,CondatVu(abstol=1e-8,reltol=1e-7),options)
@assert res.converged
@assert res.r_norm <= res.eps_pri && res.s_norm <= res.eps_dual
println(res)

# --- Warm start the primal and dual ---
# Copy x and y to preserve the preceding fit; otherwise solve! overwrites them.
xnext, ynext = copy(res.minimizer), copy(res.y)
next = solve!(xnext,f,NormL1(.02),h,D,CondatVu(τ=res.τ,σ=res.σ,abstol=1e-8,reltol=1e-7),options;y=ynext)
@assert next.converged
