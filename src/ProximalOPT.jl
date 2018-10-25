module ProximalOPT

include("lineSearches.jl")

using Printf
using ProximalBase
using .LineSearches

export
  # proximal minimization algorithms
  ProximalGradientDescent,
  AccProxGradDescent, ActiveAccProxGradDescent,
  solve!,

  # types
  ProximalSolver,
  ProximalOptions,
  OptimizationState, OptimizationTrace



include("types.jl")

include("solve.jl")

# Solvers
include("solvers/proximal_gradient_descent.jl")

include("utils.jl")

end
