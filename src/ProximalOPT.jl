module ProximalOPT

export
  # types
  DifferentiableFunction,
  ProximableFunction,

  # proximal minimization algorithms
  ProxGradDescent, AccProxGradDescent, ActiveAccProxGradDescent,
  ProximalSolver,
  solve!,

  # smooth functions
  QuadraticFunction, L2Loss,
  gradient, gradient!, value_and_gradient!,

  #utilities
  ProximalOptions

include("utils.jl")

# DifferentiableFunctions
include("diff_functions.jl")

# ProximableFunctions
include("proximal_functions.jl")

# Solvers
include("proximal_solvers.jl")

end
