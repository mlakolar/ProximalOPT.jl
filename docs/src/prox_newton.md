# Proximal Newton

[`ProxNewton`](@ref) fits penalized generalized linear models,

```math
F(\beta) = s\sum_i w_i\,\ell(\eta_i, y_i) + g(\beta), \qquad \eta = X\beta + o,
```

for a `GLMLoss` of ProximalBase (logistic, Poisson, Huber, or one of your own)
and a separable penalty ``g``, as glmnet does (Friedman, Hastie and Tibshirani,
2010). Each step minimizes a quadratic model of the loss plus the penalty by
coordinate descent, and a line search on ``F`` makes sure the objective
decreases. It also solves the graphical lasso, by QUIC; see
[The graphical lasso by QUIC](@ref).

## The quadratic model

Expand each term of the loss to second order in the linear predictor around
the current ``\eta``:

```math
\ell(\eta_i', y_i) \approx \ell(\eta_i, y_i) + \ell'(\eta_i, y_i)(\eta_i' - \eta_i) + \tfrac12 \ell''(\eta_i, y_i)(\eta_i' - \eta_i)^2
= \tfrac12\, \ell''(\eta_i, y_i)\,(z_i - \eta_i')^2 + \text{const},
```

with the working response ``z_i = \eta_i - \ell'(\eta_i, y_i)/\ell''(\eta_i, y_i)``.
With ``\eta' = X\beta' + o``, the model of the loss is the weighted least
squares loss

```math
\frac{s}{2}\sum_i v_i\,\big(z_i - o_i - x_i^\top\beta'\big)^2, \qquad v_i = w_i\,\ell''(\eta_i, y_i),
```

the iteratively reweighted least squares of generalized linear models, with
the curvatures floored at ``\sqrt{\varepsilon}`` so that the working responses
stay finite. For the logistic loss ``\log(1 + e^\eta) - y\eta``,
``\ell' = \sigma(\eta) - y`` and ``\ell'' = \sigma(\eta)(1 - \sigma(\eta))``,
with ``\sigma`` the logistic function; for the Poisson loss ``e^\eta - y\eta``,
``\ell' = e^\eta - y`` and ``\ell'' = e^\eta``.

## A step

1. Compute the weights ``v`` and the working responses ``z`` at the current
   ``\beta``, and update the least squares model in place.
2. Minimize the model plus ``g`` by coordinate descent (`inner`, a
   [`CDSolver`](@ref)), warm started from ``\beta``, which gives ``\hat\beta``.
   The linear predictor ``\hat\eta = X\hat\beta + o`` comes from the residuals of
   the model, without a product by ``X``.
3. Backtrack along ``d = \hat\beta - \beta``: try ``t = 1, \beta_{\text{ls}}, \beta_{\text{ls}}^2, \dots``
   and accept the first ``t`` with
   ``F(\beta + t d) \le F(\beta) + 10\,\varepsilon|F(\beta)|``, where
   ``\eta + t(\hat\eta - \eta)`` gives the loss without a product by ``X``. If
   ``t`` falls below ``10^{-10}``, the step is abandoned and ``\beta`` kept.

The method stops when ``|F(\beta) - F(\beta^+)| \le \mathrm{tol}\cdot\max\{|F(\beta^+)|, 1\}``,
or after `maxiter` steps. For a convex loss and penalty, and with the
quadratic model strongly convex, ``d`` is a descent direction of ``F`` unless
``\beta`` is a minimizer, and with exact inner solves the method converges,
superlinearly near the solution (Lee, Sun and Saunders, 2014).

## Why it is faster

Coordinate descent on the loss itself ([Losses for coordinate descent](@ref))
uses a global bound on the curvature, ``\sup\ell''``, which for the logistic
loss is ``1/4`` and is far above ``\ell''(\eta_i)`` wherever the fitted
probabilities are close to 0 or 1: its steps are short. It also evaluates
``\ell'`` at every observation in every coordinate step. The quadratic model
uses the curvature at the current point, and its coordinate steps are those of
weighted least squares, which cost ``O(n)`` multiplications without
evaluating the loss. On the logistic lasso problems of the package's
benchmark, momentum makes the inner solves 1.0 to 4.5 times faster. The
Poisson loss, whose curvature is unbounded, can only be fitted this way.

## Intercepts and paths

For a logistic or Poisson loss, `ProxNewton` takes an unpenalized intercept
through the model (`cdmodel(f, x; intercept = true)`), and [`solve_path`](@ref)
takes `ProxNewton` with the same sequential strong rule as coordinate descent.
The path also stops once the fit explains a fraction `devmax = 0.999` of the
null deviance, as glmnet's does:

```julia
yb = Float64.(y .> 0)                                    # binary responses
m = ProximalSolvers.cdmodel((1 / n) * LogisticLoss(yb, X), SparseIterate(p); intercept = true)
λmax = lambda_max(m, NormL1(1.0), SparseIterate(p))      # at the fit with only the intercept
b0 = Float64[]                                           # the intercepts along the path
path = solve_path(m, NormL1(1.0), λmax .* 10 .^ range(0, -2; length = 50), ProxNewton();
                  callback = (k, λ, β, m) -> (push!(b0, ProximalSolvers.intercept(m, β)); false))
```

## The graphical lasso by QUIC

For the graphical lasso,

```math
F(X) = w\big(\operatorname{tr}(SX) - \log\det X\big) + \sum_{ij}\lambda_{ij}\lvert X_{ij}\rvert,
```

`f` a `LogDetLoss(S, w)` (or a scaled one) and `g` a `NormL1` (a matrix of
weights can leave the diagonal free), `ProxNewton` is QUIC (Hsieh, Sustik,
Dhillon and Ravikumar, 2014), on a `SymmetricSparseIterate` or a dense
symmetric matrix. With ``W = X^{-1}``, the step ``\Delta`` minimizes the
second-order model

```math
w\Big(\tfrac12\operatorname{tr}(W\Delta W\Delta) + \operatorname{tr}\big((S - W)\Delta\big)\Big) + g(X + \Delta)
```

by coordinate descent over the free set, the entries that are nonzero or whose
gradient exceeds the penalty, with up to ``2(1 + \lfloor k/3\rfloor)`` passes
at step ``k``, and an Armijo line search keeps ``X`` positive definite. The
solution is block diagonal along the connected components of the graph of the
entries with ``w\lvert S_{ij}\rvert`` above the penalty, so each component is
solved on its own. The method stops when the relative duality gap (see
[Duality gaps](@ref)) is at most `tol`, which the `gap` of the result reports.
`X` is the starting point when it is positive definite; otherwise, zero for
example, the start is the diagonal solution ``1/(S_{ii} + \lambda_{ii}/w)``. A
Newton step costs a few Cholesky factorizations, where an iteration of
[ADMM](@ref) costs an eigendecomposition.

## Settings

`ProxNewton(; inner = CDSolver(; tol = 1e-8), maxiter = 100, tol = 1e-8, β = 0.5)`.
`solve!` takes a `SparseIterate` (a `SymmetricSparseIterate` or a `Matrix` for
the graphical lasso) and returns a [`CDResults`](@ref) whose
`passes` sum the inner passes, and which is `converged` when the outer stopping
rule was met and every inner solve converged.

## References

- J. Friedman, T. Hastie and R. Tibshirani. Regularization Paths for
  Generalized Linear Models via Coordinate Descent. *Journal of Statistical
  Software*, 33(1), 2010.
- C.-J. Hsieh, M. A. Sustik, I. S. Dhillon and P. Ravikumar. QUIC: Quadratic
  Approximation for Sparse Inverse Covariance Estimation. *Journal of Machine
  Learning Research*, 15, 2014.
- J. D. Lee, Y. Sun and M. A. Saunders. Proximal Newton-Type Methods for
  Minimizing Composite Functions. *SIAM Journal on Optimization*, 24(3), 2014.
