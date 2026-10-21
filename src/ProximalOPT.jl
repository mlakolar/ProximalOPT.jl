module ProximalOPT

include("lineSearches.jl")

using LinearAlgebra
using Printf
using ProximalBase
using .LineSearches

export
  # proximal minimization algorithms
  ProximalGradientDescent,
  AccProxGradDescent,
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
include("solvers/admm.jl")

include("utils.jl")

end
