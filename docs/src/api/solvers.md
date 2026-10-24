# Solver reference

```@docs
ProximalSolvers
solve!
ProximalSolver
ProximalOptions
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

## Coordinate descent solvers

```@docs
CDSolver
ProxNewton
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
