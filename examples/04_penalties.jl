# Coordinate descent with other separable penalties,
#
#     minimize  ‖y - Xβ‖² / (2n) + Σₖ gₖ(βₖ),
#
# reached through their scalar proximal operators, `prox_entry` of
# ProximalBase. A coordinate step minimizes the loss along βₖ, a quadratic,
# plus gₖ exactly: for the elastic net, MCP, SCAD, the nonnegative lasso and
# box constraints alike. MCP and SCAD are not convex, and coordinate descent
# converges to a stationary point (Breheny and Huang, 2011). The coordinate
# problems themselves stay convex while ‖Xₖ‖² / n > 1/a for MCP, and
# ‖Xₖ‖² / n > 1/(a - 1) for SCAD, as for standardized columns and the usual
# a = 3 and a = 3.7.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Printf

rng = Xoshiro(4)
Random.seed!(4)
n, p = 200, 500
X = randn(rng, n, p)
β_true = zeros(p)
β_true[1:5] = [3.0, -3.0, 2.0, -2.0, 1.5]
y = X * β_true + randn(rng, n)
f = (1 / n) * LeastSquaresLoss(y, X);

# --- Less bias than the lasso ---
#
# The lasso shrinks every nonzero coefficient towards zero by about λ. MCP and
# SCAD stop penalizing a coefficient beyond aλ, so large coefficients are not
# shrunk. The elastic net, λ‖β‖₁ + (λ₂/2)‖β‖², shrinks more than the lasso, but
# keeps groups of correlated columns together.

λ = 0.25
penalties = ["lasso" => NormL1(λ), "elastic net" => ElasticNet(λ, 0.1),
             "MCP" => MCP(λ, 3.0), "SCAD" => SCAD(λ, 3.7)]
fits = map(penalties) do (name, g)
    β = SparseIterate(p)
    res = solve!(β, f, g, CDSolver(; tol = 1e-10))
    b = Vector(β)
    @printf "%-12s nonzeros %3d   error ‖β - β*‖ %.3f   β[1:5] = %s\n" name count(!iszero, b) norm(b - β_true) round.(b[1:5]; digits = 2)
    name => b
end;

# --- Paths with nonconvex penalties ---
#
# `solve_path` takes any penalty with a level: `MCP(1.0, a)` stands for the
# family `MCP(λ, a)`. Warm starts along the path follow a branch of stationary
# points from zero.

λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))
λs = λmax .* 10 .^ range(0, -1.5; length = 30)
mcp_path = solve_path(f, MCP(1.0, 3.0), λs, CDSolver(; tol = 1e-10))
for k in 5:5:30
    b = mcp_path.solutions[k]
    @printf "λ / λmax = %.3f   nonzeros %2d   β[1:5] = %s\n" λs[k] / λmax count(!iszero, b) round.(Vector(b[1:5]); digits = 2)
end

# --- Continuation for nonconvex penalties ---
#
# From zero, coordinate descent on MCP or SCAD can stop at a poor stationary
# point, as here with p ≫ n and correlated columns. `continuation = K` solves
# at K levels from λmax down to the level of g first, so the solution follows a
# branch of stationary points from zero, as along a path, and ends at a much
# better one.

rng2 = Xoshiro(6)
n2, p2 = 100, 2000
X2 = randn(rng2, n2, p2)
for j in 2:p2                                   # neighbouring columns have correlation 0.8
    X2[:, j] .= 0.8 .* X2[:, j-1] .+ 0.6 .* X2[:, j]
end
β2 = zeros(p2)
β2[1:10:100] = [3.0, -3.0, 2.0, -2.0, 1.5, 1.0, -1.0, 1.0, -1.0, 1.0]
y2 = X2 * β2 + randn(rng2, n2)
f2 = (1 / n2) * LeastSquaresLoss(y2, X2)
g2 = MCP(0.05 * lambda_max(f2, NormL1(1.0), SparseIterate(p2)), 3.0)
Random.seed!(1)
βd, βc = SparseIterate(p2), SparseIterate(p2)
direct = solve!(βd, f2, g2, CDSolver(; tol = 1e-10))
continued = solve!(βc, f2, g2, CDSolver(; tol = 1e-10); continuation = 20)
@printf "from zero:        objective %.4f   error ‖β - β*‖ %.3f   passes %d\n" direct.minimum norm(Vector(βd) - β2) direct.passes
@printf "by continuation:  objective %.4f   error ‖β - β*‖ %.3f   passes %d\n" continued.minimum norm(Vector(βc) - β2) continued.passes

# --- Signs and bounds ---
#
# `NormL1Nonneg(λ)` is the lasso over β ≥ 0, for coefficients known to be
# nonnegative. The constraint removes the noise variables that the lasso
# brings in with negative coefficients.

yp = X * abs.(β_true) + randn(rng, n)
fp = (1 / n) * LeastSquaresLoss(yp, X)
βn, βl = SparseIterate(p), SparseIterate(p)
solve!(βn, fp, NormL1Nonneg(λ), CDSolver(; tol = 1e-10))
solve!(βl, fp, NormL1(λ), CDSolver(; tol = 1e-10))
(nonnegative_lasso = findall(!iszero, βn), lasso = findall(!iszero, βl))

# `IndBox(l, u)`, the indicator of the box l ≤ β ≤ u, has no level: coordinate
# descent then solves bounded least squares, here with n > p.

Xs, ys = X[:, 1:20], X[:, 1:20] * [3.0; -3.0; 0.5 * ones(18)] + randn(rng, n)
fs = (1 / n) * LeastSquaresLoss(ys, Xs)
βb = SparseIterate(20)
solve!(βb, fs, IndBox(-2.0, 2.0), CDSolver(; tol = 1e-10))
round.(Vector(βb)[1:6]; digits = 3)'

# --- Unpenalized coordinates ---
#
# A zero weight in `NormL1(λ, w)` leaves a coordinate unpenalized, here the
# last two, which have no effect on y and still enter the model.

w = ones(p)
w[end-1:end] .= 0
βw = SparseIterate(p)
solve!(βw, f, NormL1(λ, w), CDSolver(; tol = 1e-10))
findall(!iszero, βw)'

# --- Checks ---
#
# The optimality conditions of each problem, with the gradient
# ∇f(β) = X'(Xβ - y) / n. For MCP and SCAD, those of a stationary point:
# ∇ₖf(β) + gₖ'(βₖ) = 0 where βₖ ≠ 0, and |∇ₖf(β)| ≤ λ where βₖ = 0.

∇f(b) = X' * (X * b - y) / n
mcp′(t, λ, a) = sign(t) * max(λ - abs(t) / a, 0.0)
scad′(t, λ, a) = sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))
lasso, enet, mcp, scad = last.(fits)
stationary(b, d) = maximum(k -> iszero(b[k]) ? max(abs(∇f(b)[k]) - λ, 0.0) : abs(∇f(b)[k] + d(b[k])), 1:p)
@assert stationary(lasso, t -> λ * sign(t)) < 1e-8
@assert stationary(enet, t -> λ * sign(t) + 0.1t) < 1e-8
@assert stationary(mcp, t -> mcp′(t, λ, 3.0)) < 1e-8
@assert stationary(scad, t -> scad′(t, λ, 3.7)) < 1e-8
@assert norm(mcp - β_true) < norm(lasso - β_true) && norm(scad - β_true) < norm(lasso - β_true)
@assert maximum(abs, mcp_path.solutions[1]) < 1e-12        # zero, up to rounding at λmax
@assert continued.converged && continued.minimum < direct.minimum
@assert norm(Vector(βc) - β2) < norm(Vector(βd) - β2)

bn, gn = Vector(βn), X' * (X * Vector(βn) - yp) / n
@assert all(>=(0), bn) && issubset(1:5, findall(!iszero, bn))
@assert maximum(k -> bn[k] > 0 ? abs(gn[k] + λ) : max(-gn[k] - λ, 0.0), 1:p) < 1e-8

bb = Vector(βb)
gb = Xs' * (Xs * bb - ys) / n
@assert all(abs.(bb) .<= 2)
@assert maximum(k -> bb[k] == 2 ? max(gb[k], 0.0) : bb[k] == -2 ? max(-gb[k], 0.0) : abs(gb[k]), 1:20) < 1e-8
@assert bb[1] == 2 && bb[2] == -2

bw, gw = Vector(βw), ∇f(Vector(βw))
@assert all(!iszero, bw[end-1:end])
@assert maximum(abs, gw[end-1:end]) < 1e-8
