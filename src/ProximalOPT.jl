module ProximalOPT

using ProximalBase

export
  # proximal minimization algorithms
  ProxGradDescent,
  # AccProxGradDescent, ActiveAccProxGradDescent,
  solve!,

  # types
  ProximalSolver,
  ProximalOptions,
  OptimizationState,OptimizationTrace



include("types.jl")
include("utils.jl")

include("solve.jl")

# Solvers
include("solvers/proximal_gradient_descent.jl")

end
