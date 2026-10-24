####################################
#
#  Precompilation
#
####################################

# The solves that users and the dependent packages run most, compiled into the
# package image so that their first call does not pay for compilation.
# PrecompileTools runs this workload when the package is precompiled; the
# preference `precompile_workload = false` skips it.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
  n, p = 40, 12
  X = [cos(1.0 * (i + 3j)) + 0.1 * sin(1.0 * i * j) for i in 1:n, j in 1:p]
  β = [j <= 3 ? 1.0 : 0.0 for j in 1:p]
  y = X * β .+ [0.1 * sin(2.0 * i) for i in 1:n]
  yb = Float64.(y .> 0)
  A, b = X'X / n, -X'y / n
  S = A + I                              # symmetric positive definite
  groups = [1:4, 5:8, 9:12]
  @compile_workload begin
    f = (1 / n) * LeastSquaresLoss(y, X)
    λ = 0.1 * lambda_max(f, NormL1(1.0), SparseIterate(p))
    # coordinate descent: the lasso in random order with momentum, cyclic, with
    # an intercept, along a path, by continuation, reweighted and refitted
    solve!(SparseIterate(p), f, NormL1(λ), CDSolver())
    solve!(SparseIterate(p), f, NormL1(λ), CDSolver(; order = :cyclic))
    x = SparseIterate(p)
    solve!(x, cdmodel(f, x; intercept = true), NormL1(λ), CDSolver())
    solve_path(f, NormL1(1.0), [10λ, 3λ, λ], CDSolver())
    solve!(SparseIterate(p), f, NormL1(λ), CDSolver(); continuation = 3)
    reweight!(SparseIterate(p), f, NormL1(λ), (g, x, m) -> (g, 0.0), CDSolver(); maxiter = 1)
    x = SparseIterate(p)
    solve!(x, f, NormL1(λ), CDSolver())
    refit!(x, f)
    # other losses and penalties
    solve!(SparseIterate(p), GLMLoss(ProximalBase.Squared(), y, X; weights = fill(1.0 / n, n)), NormL1(λ), CDSolver())
    solve!(SparseIterate(p), (1 / n) * LogisticLoss(yb, X), NormL1(0.01), CDSolver())
    solve!(SparseIterate(p), (1 / n) * LogisticLoss(yb, X), NormL1(0.01), ProxNewton())
    solve!(SparseIterate(p), QuadraticFunction(A, b), NormL1(λ), CDSolver())
    solve!(SparseIterate(p), f, ElasticNet(λ, 0.1), CDSolver())
    solve!(SparseIterate(p), SqrtLeastSquaresLoss(y, X), NormL1(0.5), CDSolver())
    solve!(SparseIterate(p), SqrtQuadraticFunction(A, b, sum(abs2, y) / n), NormL1(0.1), CDSolver())
    g = GroupNormL2(λ, groups)
    solve!(AtomIterate(Float64, p, groups), f, g, CDSolver())
    solve!(SymmetricSparseIterate(p), KronQuadratic(S, S, S - I), NormL1(0.1), CDSolver())
    # the gradient methods and ADMM
    solve!(zeros(p), f, NormL1(λ), ProximalGradientDescent())
    solve!(zeros(p), f, NormL1(λ), AccProxGradDescent())
    solve!(SparseIterate(p), f, NormL1(λ), ActiveAccProxGradDescent())
    solve!(Matrix(1.0I, p, p), LogDetLoss(S), NormL1(0.1), ADMMSolver())
  end
end
