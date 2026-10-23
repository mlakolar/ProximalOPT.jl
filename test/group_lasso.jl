module GroupLassoTest

using Test
using Random
using LinearAlgebra
using ProximalBase
using ProximalSolvers

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
g = GroupNormL2(λ, groups)

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
    @test ProximalSolvers.converged(res)
    @test res.minimum ≈ f(x) + g(x)
    @test kkt_residual(x, A * x + b, groups, λ * w) < 1e-5
    @test x ≈ x_ref atol = 1e-5

    # the groups of the model are selected, and most others are zero
    @test issubset(model, nonzero_groups(x, groups))
    @test length(nonzero_groups(x, groups)) < 15

    # ‖Y - Xx‖² / 2 = n f(x) + const, so the penalty scales by n
    x = zeros(p)
    res = solve!(x, LeastSquaresLoss(Y, X), GroupNormL2(n * λ, groups), solver, opt)
    @test ProximalSolvers.converged(res)
    @test x ≈ x_ref atol = 1e-5
  end

  @testset "zero solution" begin
    # λ ≥ λmax: every solver stays at x = 0. Just above λmax, since at λmax
    # itself rounding, which depends on the BLAS of the machine, decides whether
    # the group that attains it violates the optimality condition
    λ0 = λmax * (1 + 1e-9)
    for solver in solvers
      x = zeros(p)
      solve!(x, f, GroupNormL2(λ0, groups), solver, opt)
      @test x == zeros(p)
    end
    x = zeros(p)
    res = solve!(x, f, GroupNormL2(λ0, groups), ActiveAccProxGradDescent(), opt)
    @test res.outer_iterations == 0

    # just below λmax, the group that attains λmax enters
    x = zeros(p)
    solve!(x, f, GroupNormL2(0.99 * λmax, groups), AccProxGradDescent(), opt)
    kmax = argmax([norm(b[G]) / w[k] for (k, G) in enumerate(groups)])
    @test nonzero_groups(x, groups) == [kmax]
  end

  @testset "groups of one coordinate give the lasso" begin
    singletons = [[i] for i in 1:p]
    for solver in (AccProxGradDescent(), ActiveAccProxGradDescent())
      x1, x2 = zeros(p), zeros(p)
      solve!(x1, f, GroupNormL2(λ, singletons), solver, opt)
      solve!(x2, f, NormL1(λ), solver, opt)
      @test x1 ≈ x2 atol = 1e-5
    end
  end

  @testset "unpenalized coordinates" begin
    # the coordinates of the first group (a single one) and of the fifth are in no group
    free = [groups[1]; groups[5]]
    penalized = groups[[2:4; 6:end]]
    gu = GroupNormL2(λ, penalized)
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

  @testset "AtomIterate" begin
    active_groups(x) = sort!([activecoordinates(x)[a] for a in 1:nactive(x)])

    # the gradient solvers keep the iterate an AtomIterate, whose active groups
    # are the nonzero ones at the end
    for solver in (ProximalGradientDescent(), AccProxGradDescent())
      x = AtomIterate(p, g.groups)
      res = solve!(x, f, g, solver, opt)
      @test res.minimizer === x
      @test ProximalSolvers.converged(res)
      @test Vector(x) ≈ x_ref atol = 1e-5
      @test active_groups(x) == nonzero_groups(x, groups)
    end

    # the active-set solver adds and removes whole groups
    x = AtomIterate(p, g.groups)
    res = solve!(x, f, g, ActiveAccProxGradDescent(), opt)
    @test res.minimizer === x
    @test res.converged
    @test res.max_violation <= 0
    @test Vector(x) ≈ x_ref atol = 1e-5
    @test active_groups(x) == nonzero_groups(x, groups)
    @test res.active_set_size == sum(length, groups[active_groups(x)])

    # a dense starting point takes the same path
    xd = zeros(p)
    resd = solve!(xd, f, g, ActiveAccProxGradDescent(), opt)
    @test xd == Vector(x)
    @test resd.outer_iterations == res.outer_iterations
    @test resd.active_set_size == res.active_set_size

    # one group per outer iteration; the inner solves stop at the tolerances, so
    # the solution is as accurate as in active_set.jl
    x1 = AtomIterate(p, g.groups)
    res1 = solve!(x1, f, g, ActiveAccProxGradDescent(; max_add = 1, growth = 1), opt)
    @test res1.converged
    @test res1.outer_iterations >= nactive(x1)
    @test Vector(x1) ≈ x_ref atol = 1e-4

    # on a SparseIterate, the coordinates of a group at zero have the same
    # violation and join together, though max_add is smaller than the group
    xs = SparseIterate(p)
    ress = solve!(xs, f, g, ActiveAccProxGradDescent(; max_add = 1, growth = 1), opt)
    @test ress.converged
    @test ress.outer_iterations < 20
    @test Vector(xs) ≈ x_ref atol = 1e-4

    # with a separable penalty, any groups serve: a group at zero joins when one
    # of its coordinates violates the optimality conditions
    xl, xl_ref = AtomIterate(p, groups), zeros(p)
    solve!(xl_ref, f, NormL1(λ), AccProxGradDescent(), opt)
    res = solve!(xl, f, NormL1(λ), ActiveAccProxGradDescent(), opt)
    @test res.converged
    @test Vector(xl) ≈ xl_ref atol = 1e-5

    # for the group lasso the groups of x must be those of g
    @test_throws ArgumentError solve!(AtomIterate(p, 10), f, g, ActiveAccProxGradDescent(), opt)
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
    res1 = solve!(Θ1, LogDetLoss(S), GroupNormL2(0.1, pairs, ones(length(pairs))),
                  ADMMSolver(), ProximalOptions(; maxiter = 5000))
    res2 = solve!(Θ2, LogDetLoss(S), NormL1(0.1, (ones(q, q) - I) / √2),
                  ADMMSolver(), ProximalOptions(; maxiter = 5000))
    @test res1.converged && res2.converged
    @test Θ1 ≈ Θ2 rtol = 1e-8
    @test count(iszero, Θ1) > 0
  end

end

end
