"""
    ProximalOPT

Solvers for `minimize f(x) + g(x)`, with `f` smooth and `g` proximable, built
on the functions and proximal operators of ProximalBase:
[`ProximalGradientDescent`](@ref), [`AccProxGradDescent`](@ref),
[`ActiveAccProxGradDescent`](@ref) and [`ADMMSolver`](@ref), all called
through [`solve!`](@ref) with [`ProximalOptions`](@ref).
"""
module ProximalOPT

include("lineSearches.jl")

using LinearAlgebra
using Printf
using ProximalBase
using SparseArrays: dropzeros!
using .LineSearches

export
  # proximal minimization algorithms
  ProximalGradientDescent,
  AccProxGradDescent,
  ActiveAccProxGradDescent,
  ADMMSolver,
  solve!,

  # types
  ProximalSolver,
  ProximalOptions,
  OptimizationState, OptimizationTrace



include("types.jl")

include("solve.jl")

# Solvers
include("solvers/proximal_gradient_descent.jl")
include("solvers/accelerated_proximal_gradient_descent.jl")
include("solvers/active_accelerated_proximal_gradient_descent.jl")
include("solvers/admm.jl")

include("utils.jl")

end
