module ProximalOPT

using ProximalBase

export
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

# Solvers
include("proximal_solvers.jl")

end
