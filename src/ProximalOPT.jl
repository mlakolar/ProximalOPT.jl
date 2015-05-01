module ProximalOPT

using Compat

export
  # types
  DifferentiableFunction,
  ProximableFunction,

  # proximal minimization algorithms
  ProxGradDescent, AccProxGradDescent, ProximalSolver,
  solve!,

  # smooth functions
  QuadraticFunction, L2Loss,
  gradient, gradient!, value_and_gradient!,

  # proximal functions
  ProxZero,
  ProxL1, ProxL2, ProxL2Sq, ProxNuclear, ProxSumProx,
  ProxL1L2, ProxL1Nuclear,
  value, prox!,

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
