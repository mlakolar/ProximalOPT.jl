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
import ProximalBase: gradient!, value_and_gradient!
using SparseArrays: dropzeros!, sparse, sparsevec
using .LineSearches

export
  # proximal minimization algorithms
  ProximalGradientDescent,
  AccProxGradDescent,
  ActiveAccProxGradDescent,
  ADMMSolver,
  CDSolver,
  ProxNewton,
  solve!,

  # coordinate descent: losses, models, penalty levels, drivers
  KronQuadratic,
  SqrtLeastSquaresLoss,
  CDModel, cdmodel,
  lambda_max, solve_path,
  reweight!, refit!,

  # types
  ProximalSolver,
  ProximalOptions,
  OptimizationState, OptimizationTrace,
  CDResults, PathResults, ReweightResults



include("types.jl")

include("solve.jl")

# Solvers
include("solvers/proximal_gradient_descent.jl")
include("solvers/accelerated_proximal_gradient_descent.jl")
include("solvers/active_accelerated_proximal_gradient_descent.jl")
include("solvers/admm.jl")

# Coordinate descent
include("cd/models.jl")
include("cd/solver.jl")
include("cd/leastsquares.jl")
include("cd/quadratic.jl")
include("cd/kron.jl")
include("cd/sqrtlasso.jl")
include("cd/glm.jl")
include("cd/path.jl")
include("cd/drivers.jl")

include("utils.jl")

end
