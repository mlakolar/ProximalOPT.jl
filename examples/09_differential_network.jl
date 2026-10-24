# The difference of two precision matrices, Δ = Ωx - Ωy, estimated directly
# from two sample covariance matrices by the D-trace loss of Zhao, Cai and Li
# (2014):
#
#     minimize  tr(Sx Δ Sy Δ) / 2 - tr(Δ (Sy - Sx)) + λ ‖Δ‖₁   over symmetric Δ.
#
# The loss is `KronQuadratic(Sx, Sy, Sx - Sy)`, ⟨Δ, SxΔSy⟩/2 + ⟨Sx - Sy, Δ⟩. Its
# gradient over symmetric matrices, (SxΔSy + SyΔSx)/2 + Sx - Sy, vanishes at
# Δ = Sx⁻¹ - Sy⁻¹, so the penalized minimizer estimates Ωx - Ωy without
# estimating either precision matrix: Δ can be sparse when they are not.
#
# Coordinate descent runs on a `SymmetricSparseIterate`, whose coordinates are
# the entries of the lower triangle; a step on an off-diagonal coordinate moves
# Δ[r, c] and Δ[c, r] together. The model caches M = SxΔSy, so a step costs
# O(p²).

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics, Printf

rng = Xoshiro(9)
Random.seed!(9)
p, n = 30, 500
Ωx = Matrix(SymTridiagonal(fill(1.0, p), fill(0.4, p - 1)))
Δ_true = zeros(p, p)
for (r, c, v) in ((2, 1, 0.4), (12, 11, 0.4), (20, 5, 0.3), (25, 15, -0.3))
    Δ_true[r, c] = Δ_true[c, r] = v
end
Ωy = Ωx - Δ_true
@assert isposdef(Ωy)
draws(Ω, m) = randn(rng, m, p) / cholesky(Ω).U'      # m draws from N(0, Ω⁻¹)
Sx = cov(draws(Ωx, n); corrected = false)
Sy = cov(draws(Ωy, n); corrected = false)

f = KronQuadratic(Sx, Sy, Sx - Sy);

# --- A path ---
#
# At zero the gradient is Sx - Sy, so λmax = maxᵢⱼ |Sx - Sy|ᵢⱼ. `solve_path`
# needs the iterate, here a zero `SymmetricSparseIterate`.

λmax = lambda_max(f, NormL1(1.0), SymmetricSparseIterate(p))
λs = λmax .* 10 .^ range(0, -1; length = 15)
path = solve_path(f, NormL1(1.0), λs, CDSolver(; tol = 1e-10); x = SymmetricSparseIterate(p))
edges(D) = count(c -> c[1] > c[2] && D[c] != 0, CartesianIndices(D))
for k in 1:2:15
    D = Matrix(path.solutions[k])
    @printf "λ / λmax = %.3f   edges %3d   of the 4 true ones %d   passes %d\n" λs[k] / λmax edges(D) count(c -> c[1] > c[2] && D[c] != 0 && Δ_true[c] != 0, CartesianIndices(D)) path.passes[k]
end

# --- One level, with momentum, then a refit ---
#
# Momentum is off by default for this loss, and `momentum = true` turns it on.
# `refit!` then minimizes the loss on the support: the gradient vanishes there,
# and the estimates lose the shrinkage of the penalty. Each row below is
# (r, c, true value, penalized estimate, refitted estimate):

λ = 0.3λmax
Δ = SymmetricSparseIterate(p)
res = solve!(Δ, f, NormL1(λ), CDSolver(; tol = 1e-10, momentum = true))
D = Matrix(Δ)
support = [(c[1], c[2]) for c in CartesianIndices(D) if c[1] > c[2] && D[c] != 0]
refit!(Δ, f)
Dr = Matrix(Δ)
[(r, c, round(Δ_true[r, c]; digits = 2), round(D[r, c]; digits = 2), round(Dr[r, c]; digits = 2)) for (r, c) in support]

# --- Checks ---
#
# The optimality conditions: with G = (SxΔSy + SyΔSx)/2 + Sx - Sy, Gᵢⱼ = -λ sign(Δᵢⱼ)
# where Δᵢⱼ ≠ 0, and |Gᵢⱼ| ≤ λ where Δᵢⱼ = 0. After the refit, G = 0 on the
# support.

G(D) = (Sx * D * Sy + Sy * D * Sx) / 2 + Sx - Sy
kkt(D, λ) = maximum(c -> D[c] == 0 ? max(abs(G(D)[c]) - λ, 0.0) : abs(G(D)[c] + λ * sign(D[c])), CartesianIndices(D))
@assert all(kkt(Matrix(path.solutions[k]), λs[k]) < 1e-8 for k in eachindex(λs))
@assert kkt(D, λ) < 1e-8
@assert issymmetric(D) && issymmetric(Dr)
@assert maximum(abs, G(Dr)[Dr .!= 0]) < 1e-10
@assert all(D[r, c] != 0 for (r, c) in ((2, 1), (12, 11), (20, 5), (25, 15)))
