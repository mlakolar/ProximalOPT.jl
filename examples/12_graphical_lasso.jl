# The graphical lasso: a sparse estimate of the precision matrix Ω = Σ⁻¹ of a
# Gaussian vector, whose zeros are the missing edges of its graphical model,
#
#     minimize  tr(SΘ) - log det Θ + λ Σᵢ≠ⱼ |Θᵢⱼ|    over Θ ≻ 0,
#
# by the alternating direction method of multipliers. ADMM splits Θ = Z and
# alternates the proximal operators of the two terms: that of
# tr(SΘ) - log det Θ, from an eigendecomposition, and soft thresholding. The
# returned estimate is Z, which is exactly sparse.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Statistics, Printf

rng = Xoshiro(12)
p, n = 30, 300
Ω = Matrix(SymTridiagonal(fill(1.0, p), fill(0.4, p - 1)))
Zs = randn(rng, n, p) / cholesky(Ω).U'                 # n draws from N(0, Ω⁻¹)
S = cov(Zs; corrected = false)

f = LogDetLoss(S)                                       # tr(SΘ) - log det Θ
W = ones(p, p) - I                                      # leave the diagonal unpenalized
λ = 0.1
g = NormL1(λ, W)
tight = ProximalOptions(; maxiter = 10_000);

# --- Three versions of ADMM ---
#
# The penalty parameter ρ sets the balance between the primal residual
# ‖Θ - Z‖ and the dual residual ρ‖Z - Z'‖, and a good value depends on the
# problem. Here it starts from a poor one, ρ = 20. With `adaptive = true`, ρ is
# doubled or halved whenever one residual, relative to its tolerance, exceeds
# ten times the other. Over-relaxation, α between 1.5 and 1.8, often converges
# faster still.

versions = ["ρ = 20" => ADMMSolver(; ρ = 20.0, abstol = 1e-8, reltol = 1e-6),
            "ρ = 20, adaptive" => ADMMSolver(; ρ = 20.0, abstol = 1e-8, reltol = 1e-6, adaptive = true),
            "ρ = 20, adaptive, α = 1.6" => ADMMSolver(; ρ = 20.0, α = 1.6, abstol = 1e-8, reltol = 1e-6, adaptive = true)]
fits = map(versions) do (name, solver)
    Θ = Matrix(1.0I, p, p)
    res = solve!(Θ, f, g, solver, tight)
    @printf "%-26s %4d iterations   final ρ %6.3f   objective %.10f\n" name res.iterations res.ρ res.minimum
    res
end;

# --- A path of penalty levels with warm starts ---
#
# A previous result warm starts the next solve: its estimate as the starting
# point, and its scaled dual variable `u` and final ρ, which `solve!` takes
# as a keyword and as a solver setting. Along a path of λ, each solve then
# starts close to its solution, and takes fewer iterations than a solve from
# the identity.

Θ = Matrix(1.0I, p, p)
res = nothing
for λk in (0.4, 0.3, 0.2, 0.15, 0.1)
    global res
    warm = res === nothing ? (;) : (; u = res.u)
    ρ = res === nothing ? 1.0 : res.ρ
    solver = ADMMSolver(; ρ, α = 1.6, abstol = 1e-8, reltol = 1e-6, adaptive = true)
    res = solve!(Θ, f, NormL1(λk, W), solver, tight; warm...)
    edges = count(c -> c[1] > c[2] && Θ[c] != 0, CartesianIndices(Θ))
    @printf "λ = %.2f   %4d iterations   edges %3d (true edges %d)\n" λk res.iterations edges p - 1
end

# --- Checks ---
#
# The optimality conditions of the graphical lasso, with G = S - Θ⁻¹: Gᵢᵢ = 0,
# Gᵢⱼ = -λ sign(Θᵢⱼ) where Θᵢⱼ ≠ 0, and |Gᵢⱼ| ≤ λ where Θᵢⱼ = 0, up to the
# tolerances of ADMM.

function kkt(Θ, λ)
    G = S - inv(Θ)
    maximum(CartesianIndices(Θ)) do c
        c[1] == c[2] ? abs(G[c]) : Θ[c] == 0 ? max(abs(G[c]) - λ, 0.0) : abs(G[c] + λ * sign(Θ[c]))
    end
end
@assert all(r -> r.converged, fits)
@assert all(r -> kkt(r.minimizer, λ) < 1e-4, fits)
@assert maximum(r -> abs(r.minimum - fits[1].minimum), fits) < 1e-6
@assert res.converged && kkt(Θ, 0.1) < 1e-4 && norm(Θ - fits[1].minimizer, Inf) < 1e-4
@assert issymmetric(Θ) && isposdef(Θ)
