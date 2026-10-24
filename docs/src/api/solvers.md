# Solver reference

```@docs
ProximalSolvers
solve!
ProximalSolver
ProximalOptions
SolverWorkspace
reinit!
```

## Coordinate descent solvers

```@docs
CDSolver
ProxNewton
```

## Proximal gradient solvers

```@docs
ProximalGradientDescent
AccProxGradDescent
MonotoneAccProxGradDescent
ActiveAccProxGradDescent
ProximalSolvers.LineSearches.BackTracking
```

## The ADMM solver

```@docs
ADMMSolver
```

## Duality gaps and optimality

```@docs
duality_gap
kkt_residual
```

## Result types

```@docs
ProximalSolvers.OptimizationResults
ProximalSolvers.ActiveSetResults
ProximalSolvers.ADMMResults
CDResults
OptimizationState
OptimizationTrace
```
