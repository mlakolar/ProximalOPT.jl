# Paths, λmax and screening for the sparse group lasso SparseGroupNormL2 at a
# fixed mix α, checked against the optimality conditions computed from dense
# algebra.

module SGLPathTests

using Test, LinearAlgebra, Random
using ProximalBase
using ProximalSolvers

soft(v, t) = sign.(v) .* max.(abs.(v) .- t, 0)

# the largest violation of the optimality conditions of
# f + λα Σ|xᵢ| + λ(1 - α) Σ wₖ‖x_Gₖ‖ at x, for the gradient g of f
function kkt_sgl(g, x, groups, w, λ, α)
  v = 0.0
  for (k, G) in enumerate(groups)
    xG, gG = x[G], g[G]
    if all(iszero, xG)
      v = max(v, norm(soft(gG, λ * α)) - λ * (1 - α) * w[k])
    else
      nG = norm(xG)
      for (i, xi) in enumerate(xG)
        r = xi == 0 ? max(abs(gG[i]) - λ * α, 0.0) : abs(gG[i] + λ * α * sign(xi) + λ * (1 - α) * w[k] * xi / nG)
        v = max(v, r)
      end
    end
  end
  v
end

lsgrad(X, y, x) = X' * (X * x - y) / size(X, 1)

rng = Xoshiro(11)
n, p = 120, 40
X = randn(rng, n, p)
β = zeros(p); β[1:3] .= [1.0, -1.0, 0.5]; β[11:12] .= [0.8, -0.6]
y = X * β .+ 0.3 .* randn(rng, n)
groups = [collect(1:5), collect(6:10), collect(11:20), collect(21:40)]
w = sqrt.(length.(groups))
f = (1 / n) * LeastSquaresLoss(y, X)

@testset "λmax of the sparse group lasso, α = $α" for α in (0.2, 0.5, 0.95)
  g = SparseGroupNormL2(α, 1 - α, groups)
  λmax = lambda_max(f, g, AtomIterate(p, groups))
  # the root of ‖soft(∇_G f(0), λα)‖ = λ(1 - α)wₖ, largest over the groups
  g0 = lsgrad(X, y, zeros(p))
  for (k, G) in enumerate(groups)
    r = ProximalSolvers._sglroot(g0[G], α, w[k])
    @test norm(soft(g0[G], r * α)) ≈ r * (1 - α) * w[k] rtol = 1e-10
  end
  @test λmax ≈ maximum(ProximalSolvers._sglroot(g0[G], α, w[k]) for (k, G) in enumerate(groups))
  x = AtomIterate(p, groups)
  solve!(x, f, SparseGroupNormL2(λmax * (1 + 1e-9) * α, λmax * (1 + 1e-9) * (1 - α), groups), CDSolver(; tol = 1e-12))
  @test all(iszero, Vector(x))
  x = AtomIterate(p, groups)
  solve!(x, f, SparseGroupNormL2(λmax * (1 - 1e-3) * α, λmax * (1 - 1e-3) * (1 - α), groups), CDSolver(; tol = 1e-12))
  @test any(!iszero, Vector(x))
end

@testset "paths of the sparse group lasso, α = $α" for α in (0.2, 0.5, 0.95)
  g = SparseGroupNormL2(α, 1 - α, groups)
  λmax = lambda_max(f, g, AtomIterate(p, groups))
  λs = λmax * (1 + 1e-9) .* 10 .^ range(0, -2; length = 12)
  paths = map((true, false)) do screening
    path = solve_path(f, g, λs, CDSolver(; tol = 1e-12); x = AtomIterate(p, groups), screening)
    @test path.λ == λs && all(path.converged)
    @test screening ? all(>=(1), path.kkt_checks) : all(iszero, path.kkt_checks)
    for k in eachindex(λs)
      b = Vector(path.solutions[k])
      @test kkt_sgl(lsgrad(X, y, b), b, groups, w, λs[k], α) < 1e-7
    end
    @test all(iszero, Vector(path.solutions[1]))
    path
  end
  @test paths[1].values ≈ paths[2].values rtol = 1e-10
  # continuation to one level
  λ = 0.05λmax
  x = AtomIterate(p, groups)
  r = solve!(x, f, SparseGroupNormL2(λ * α, λ * (1 - α), groups), CDSolver(; tol = 1e-12); continuation = 8)
  @test r.converged && r.kkt_checks >= 9
  @test kkt_sgl(lsgrad(X, y, Vector(x)), Vector(x), groups, w, λ, α) < 1e-7
end

@testset "the limits: the lasso (α = 1) and the group lasso (λ1 = 0)" begin
  λs = 10 .^ range(-0.2, -1.5; length = 8)
  a = solve_path(f, SparseGroupNormL2(1.0, 0.0, groups), λs, CDSolver(; tol = 1e-12); x = AtomIterate(p, groups))
  b = solve_path(f, NormL1(1.0), λs, CDSolver(; tol = 1e-12))
  @test a.values ≈ b.values rtol = 1e-9
  c = solve_path(f, SparseGroupNormL2(0.0, 1.0, groups), λs, CDSolver(; tol = 1e-12); x = AtomIterate(p, groups))
  d = solve_path(f, GroupNormL2(1.0, groups, w), λs, CDSolver(; tol = 1e-12); x = AtomIterate(p, groups))
  @test c.values ≈ d.values rtol = 1e-9
  @test lambda_max(f, SparseGroupNormL2(0.0, 1.0, groups), AtomIterate(p, groups)) ≈
        lambda_max(f, GroupNormL2(1.0, groups, w), AtomIterate(p, groups))
  # a group without any penalty (λ1 = 0, weight zero) is kept by screening and left out of λmax
  w0 = copy(w); w0[4] = 0
  g0 = SparseGroupNormL2(0.0, 1.0, groups, w0)
  lm = lambda_max(f, g0, AtomIterate(p, groups))
  @test lm ≈ lambda_max(f, GroupNormL2(1.0, groups, w0), AtomIterate(p, groups))
  path = solve_path(f, g0, lm .* [1.0, 0.5], CDSolver(; tol = 1e-12); x = AtomIterate(p, groups))
  @test any(!iszero, Vector(path.solutions[1])[groups[4]])
  @test_throws ArgumentError solve_path(f, SparseGroupNormL2(0.0, 0.0, groups), [1.0]; x = AtomIterate(p, groups))
end

end
