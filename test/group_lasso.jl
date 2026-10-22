module GroupLassoTest

using Test
using Random
using LinearAlgebra
using ProximalBase
using ProximalOPT

Random.seed!(1)

# group lasso with 60 groups of sizes 1 to 6, four of them in the model:
# f(x) = ‖Y - Xx‖² / (2n) = x'Ax/2 + b'x + const and g(x) = λ Σₖ √|Gₖ| ‖x[Gₖ]‖₂
sizes = repeat(1:6, 10)
groups = [(e - s + 1):e for (s, e) in zip(sizes, cumsum(sizes))]
n, p = 150, sum(sizes)
X = randn(n, p)
β = zeros(p)
model = [2, 9, 23, 40]
for k in model
  β[groups[k]] = randn(sizes[k]) .+ 2 .* sign.(randn(sizes[k]))
end
Y = X * β + 0.1 * randn(n)
A, b = X'X / n, -X'Y / n
f = QuadraticFunction(A, b)
w = sqrt.(sizes)
λmax = maximum(norm(b[groups[k]]) / w[k] for k in eachindex(groups))   # x = 0 is optimal for λ ≥ λmax
λ = 0.2 * λmax
g = ProxGroupL2(λ, groups)

opt = ProximalOptions(; ftol = 1e-12, xtol = 1e-12, gradtol = 1e-10, maxiter = 10000)

# largest violation of the optimality conditions of f + g at x, with gradient
# grad, for the groups with thresholds λw and the unpenalized coordinates free:
# grad[G] = -λw x[G] / ‖x[G]‖ for a nonzero group, ‖grad[G]‖ ≤ λw for a zero one
function kkt_residual(x, grad, groups, λw, free = Int[])
  r = isempty(free) ? 0. : norm(grad[free], Inf)
  for k in eachindex(groups)
    G = groups[k]
    nx = norm(x[G])
    r = max(r, iszero(nx) ? norm(grad[G]) - λw[k] : norm(grad[G] + λw[k] * x[G] / nx))
  end
  r
end
nonzero_groups(x, groups) = findall(G -> !iszero(x[G]), groups)

solvers = (ProximalGradientDescent(), AccProxGradDescent(), ActiveAccProxGradDescent())

x_ref = zeros(p)
solve!(x_ref, f, g, AccProxGradDescent(), opt)

@testset "group lasso" begin

  @testset "$(summary(solver))" for solver in solvers
    x = zeros(p)
    res = solve!(x, f, g, solver, opt)
    @test ProximalOPT.converged(res)
    @test res.minimum ≈ value(f, x) + value(g, x)
    @test kkt_residual(x, A * x + b, groups, λ * w) < 1e-5
    @test x ≈ x_ref atol = 1e-5

    # the groups of the model are selected, and most others are zero
    @test issubset(model, nonzero_groups(x, groups))
    @test length(nonzero_groups(x, groups)) < 15

    # ‖Y - Xx‖² / 2 = n f(x) + const, so the penalty scales by n
    x = zeros(p)
    res = solve!(x, LeastSquaresLoss(Y, X), ProxGroupL2(n * λ, groups), solver, opt)
    @test ProximalOPT.converged(res)
    @test x ≈ x_ref atol = 1e-5
  end

  @testset "zero solution" begin
    # λ ≥ λmax: every solver stays at x = 0
    for solver in solvers
      x = zeros(p)
      solve!(x, f, ProxGroupL2(λmax, groups), solver, opt)
      @test x == zeros(p)
    end
    x = zeros(p)
    res = solve!(x, f, ProxGroupL2(λmax, groups), ActiveAccProxGradDescent(), opt)
    @test res.outer_iterations == 0

    # just below λmax, the group that attains λmax enters
    x = zeros(p)
    solve!(x, f, ProxGroupL2(0.99 * λmax, groups), AccProxGradDescent(), opt)
    kmax = argmax([norm(b[G]) / w[k] for (k, G) in enumerate(groups)])
    @test nonzero_groups(x, groups) == [kmax]
  end

  @testset "groups of one coordinate give the lasso" begin
    singletons = [[i] for i in 1:p]
    for solver in (AccProxGradDescent(), ActiveAccProxGradDescent())
      x1, x2 = zeros(p), zeros(p)
      solve!(x1, f, ProxGroupL2(λ, singletons), solver, opt)
      solve!(x2, f, ProxL1(λ), solver, opt)
      @test x1 ≈ x2 atol = 1e-5
    end
  end

  @testset "unpenalized coordinates" begin
    # the coordinates of the first group (a single one) and of the fifth are in no group
    free = [groups[1]; groups[5]]
    penalized = groups[[2:4; 6:end]]
    gu = ProxGroupL2(λ, penalized)
    xs = map(solvers) do solver
      x = zeros(p)
      solve!(x, f, gu, solver, opt)
      @test kkt_residual(x, A * x + b, penalized, λ * sqrt.(length.(penalized)), free) < 1e-5
      x
    end
    @test xs[2] ≈ xs[1] atol = 1e-5
    @test xs[3] ≈ xs[1] atol = 1e-5
  end

  @testset "zero column in a nonzero group" begin
    # x[i] for a zero column X[:, i] stays at zero; the active-set solver must
    # not add it back after dropping it, though the rest of its group is nonzero
    i = groups[23][1]
    X0 = copy(X)
    X0[:, i] .= 0
    f0 = QuadraticFunction(X0'X0 / n, -X0'Y / n)
    x_full = zeros(p)
    solve!(x_full, f0, g, AccProxGradDescent(), opt)

    x = SparseIterate(p)
    res = solve!(x, f0, g, ActiveAccProxGradDescent(), opt)
    @test res.converged
    @test res.outer_iterations <= 3
    @test x[i] == 0
    @test 23 in nonzero_groups(Vector(x), groups)
    @test Vector(x) ≈ x_full atol = 1e-5
  end

  @testset "ADMM: graphical lasso with grouped entries" begin
    # grouping each off-diagonal pair {Θᵢⱼ, Θⱼᵢ} gives √2 |Θᵢⱼ| for a symmetric Θ,
    # the ℓ1 penalty with weights 1/√2 off the diagonal; the diagonal is in no group
    q = 8
    Z = randn(100, q)
    S = Symmetric(Z'Z / 100)
    L = LinearIndices((q, q))
    pairs = [[L[i, j], L[j, i]] for j in 1:q for i in (j + 1):q]
    Θ1, Θ2 = Matrix(1.0I, q, q), Matrix(1.0I, q, q)
    res1 = solve!(Θ1, ProxGaussLikelihood(S), ProxGroupL2(0.1, pairs, ones(length(pairs))),
                  ADMMSolver(), ProximalOptions(; maxiter = 5000))
    res2 = solve!(Θ2, ProxGaussLikelihood(S), ProxL1(0.1, (ones(q, q) - I) / √2),
                  ADMMSolver(), ProximalOptions(; maxiter = 5000))
    @test res1.converged && res2.converged
    @test Θ1 ≈ Θ2 rtol = 1e-8
    @test count(iszero, Θ1) > 0
  end

end

end
