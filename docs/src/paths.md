# Penalty paths and screening

A penalty with a level ``\lambda`` defines a family of problems, and the
solution is usually wanted along a decreasing sequence of levels, to choose
``\lambda`` by cross-validation or an information criterion.
[`solve_path`](@ref) computes it with warm starts and screening, and
`solve!(x, f, g, CDSolver(); continuation)` uses the same machinery to reach
one level.

## Penalty families

These penalties of ProximalBase have a level, and
`solve_path(f, g, λs)` replaces the level of `g` by each ``\lambda``:

| Penalty | ``g_\lambda(x)`` | weight of coordinate ``k`` |
|:--------|:-----------------|:---------------------------|
| `NormL1(λ, w)` | ``\lambda\sum_k w_k\lvert x_k\rvert`` | ``w_k`` |
| `NormL1Nonneg(λ, w)` | ``\lambda\sum_k w_k x_k`` on ``x \ge 0`` | ``w_k`` |
| `ElasticNet(λ, λ₂, w)` | ``\lambda\sum_k w_k\lvert x_k\rvert + \tfrac{\lambda_2}{2}\lVert x\rVert^2`` | ``w_k`` |
| `MCP(λ, a)`, `SCAD(λ, a)` | nonconvex, with slope ``\lambda`` at zero | ``1`` |
| `GroupNormL2(λ, groups, w)` | ``\lambda\sum_k w_k\lVert x_{G_k}\rVert_2`` | ``w_k`` |

In each, a coordinate at zero is optimal (stationary, for MCP and SCAD) while
the gradient of the loss over it is at most ``\lambda w_k`` in absolute value,
or in Euclidean norm for a group: the right derivative of every penalty at
zero is ``\lambda w_k``. A coordinate with ``w_k = 0`` is not penalized.

## The largest useful level

Zero minimizes ``f + g_\lambda`` exactly when every penalized coordinate
satisfies this condition at zero:

```math
\lambda_{\max} = \max_{k:\, w_k > 0} \frac{|\nabla_k f(0)|}{w_k}
\qquad\Big(\max_k \frac{\|\nabla_{G_k} f(0)\|_2}{w_k} \text{ for groups}\Big).
```

[`lambda_max`](@ref)`(f, g, x)` computes it from the gradient at zero, for
iterates like `x`; for the lasso of ``\|y - X\beta\|^2/(2n)`` it is
``\|X^\top y\|_\infty/n``. With unpenalized coordinates the formula still uses
the gradient at zero, while the smallest level with all penalized
coordinates at zero would use the gradient at the fit of the unpenalized
ones. A common grid is geometric, from ``\lambda_{\max}`` down to
``\epsilon\lambda_{\max}`` with ``\epsilon`` between ``10^{-3}`` and ``10^{-1}``.

## Warm starts and the strong rule

Each level starts from the solution at the level before, which is close when
the levels are close. Screening also uses it to predict which coordinates stay
at zero. Let ``c_j(\lambda) = \nabla_j f(\hat x(\lambda))`` at the solution
``\hat x(\lambda)``. The sequential strong rule of Tibshirani et al. (2012)
leaves coordinate ``j`` out at the level ``\lambda_k`` when

```math
|c_j(\lambda_{k-1})| < (2\lambda_k - \lambda_{k-1})\, w_j .
```

The rule is exact if ``|c_j(\lambda) - c_j(\lambda')| \le w_j|\lambda - \lambda'|``
along the path: then
``|c_j(\lambda_k)| < (2\lambda_k - \lambda_{k-1})w_j + (\lambda_{k-1} - \lambda_k)w_j = \lambda_k w_j``,
and coordinate ``j`` is zero at ``\lambda_k``. The condition often holds but can
fail, so the rule is checked. At each level,

1. the strong set ``S`` holds the coordinates that pass the rule's threshold,
   ``|c_j(\lambda_{k-1})| \ge (2\lambda_k - \lambda_{k-1})w_j``, those that are
   nonzero, and the unpenalized ones;
2. coordinate descent solves the problem with its full passes over ``S``
   instead of all coordinates;
3. one full gradient checks the optimality condition
   ``|\nabla_j f(x)| \le \lambda_k w_j`` of every coordinate outside ``S``; the
   violators join ``S``, and step 2 runs again from the current solution.

The gradient computed in step 3 is the one the rule needs at the next level.
`solve_path` starts the rule at the ``\lambda_{\max}`` of its starting point,
which need not be zero. The results report, per level, the passes and the
number of checks: one when the strong rule made no mistake.

## Calling `solve_path`

`solve_path(f, g, λs, method = CDSolver(); x, screening = true, maxnnz = Inf, callback = nothing)`
takes decreasing nonnegative levels. The iterate `x` is the starting point and
holds the last solution on return; it defaults to a zero `SparseIterate` when
the loss has columns, and must be given otherwise, as an `AtomIterate` for a
group penalty or a `SymmetricSparseIterate` for a symmetric matrix. The path
stops early after the first solution with more than `maxnnz` nonzero
coordinates, or when `callback(k, λ, x, m)` returns `true`; `m` is the model of
the loss, current for `x`, from which a callback can compute a validation
error, residuals or a gradient. The [`PathResults`](@ref) hold, per level, the
level, the solution as a sparse vector (a sparse matrix for a matrix iterate),
the objective, the passes, whether coordinate descent converged, and the number
of checks.

## Continuation

`solve!(x, f, g, CDSolver(); continuation = K)` sets `x` to zero and solves at
the ``K + 1`` levels

```math
\lambda_j = \lambda_{\max}\Big(\frac{\lambda}{\lambda_{\max}}\Big)^{j/K}, \qquad j = 0, 1, \dots, K,
```

geometrically from ``\lambda_{\max}`` down to the level ``\lambda`` of `g`, each
warm started from the one before and screened as above (`screening = true`,
the default except on a `SymmetricSparseIterate`, where a coordinate step
changes the gradient everywhere, so that the checks cost more passes than they
save). The levels before `g` only provide warm starts, so they are solved to
the looser tolerance `continuation_tol`, ``\sqrt{\mathrm{tol}}`` by default,
and only `g` itself to `tol`; `continuation_tol = tol` solves every level
fully. It returns the solution at `g`. When the level of `g` is zero or at
least ``\lambda_{\max}``, or `g` has no level (`IndBox`, `ZeroFunction`), it
solves from zero directly.

On a convex problem the minimizer is the same either way, and continuation is
a question of speed. On the lasso design of
[A lasso path](examples/02_lasso_path.md), it took half the passes of a direct
solve from zero, or fewer, at ``0.01\lambda_{\max}``, and fewer at
``0.02\lambda_{\max}``, but more at ``0.05\lambda_{\max}`` and above, where the
direct solve is short anyway.

For MCP and SCAD it also changes the answer. From zero, coordinate descent can
stop at a poor stationary point when ``p \gg n`` and the columns are
correlated; along the levels, the solution follows a branch of stationary
points from the zero solution, as along a path. In
[Other penalties](examples/04_penalties.md), with MCP, it ends at less than
half the objective of the direct solve, with a far smaller estimation error.

## References

- R. Tibshirani, J. Bien, J. Friedman, T. Hastie, N. Simon, J. Taylor and
  R. Tibshirani. Strong Rules for Discarding Predictors in Lasso-type Problems.
  *Journal of the Royal Statistical Society B*, 74(2), 2012.
- J. Friedman, T. Hastie and R. Tibshirani. Regularization Paths for
  Generalized Linear Models via Coordinate Descent. *Journal of Statistical
  Software*, 33(1), 2010.
