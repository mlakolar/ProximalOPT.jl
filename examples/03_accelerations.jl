# Coordinate descent needs many passes when the columns of X are strongly
# correlated and the penalty is small. `CDSolver` has two accelerations:
#
# - with the random order, the default, Nesterov momentum on the passes over
#   the active set, restarted when a pass moves against it;
# - with the cyclic order, Anderson extrapolation of the iterates on the active
#   set (`anderson = K`), kept only when it decreases the objective.
#
# On a lasso with correlated columns, all four combinations reach the same
# solution, in different numbers of passes.

using ProximalBase, ProximalSolvers, LinearAlgebra, Random, Printf

rng = Xoshiro(3)
n, p, ρ = 400, 1000, 0.9
X = randn(rng, n, p)
for j in 2:p                         # neighbouring columns have correlation ρ
    X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
end
β_true = zeros(p)
β_true[1:20:200] .= 1.0
y = X * β_true + randn(rng, n)

f = (1 / n) * LeastSquaresLoss(y, X)
λ = 0.01 * lambda_max(f, NormL1(1.0), SparseIterate(p))
g = NormL1(λ)

solvers = [
    "random order, no momentum" => CDSolver(; momentum = false),
    "random order, momentum"    => CDSolver(),
    "cyclic order"              => CDSolver(; order = :cyclic),
    "cyclic order, Anderson"    => CDSolver(; order = :cyclic, anderson = 5),
];

# --- Passes and time ---
#
# Each solver runs twice from zero, and the second run is timed: the first
# compiles. The random orders use the global random number generator, so a seed
# makes the runs repeatable.

results = map(solvers) do (name, solver)
    local β, res
    for _ in 1:2
        Random.seed!(1)
        β = SparseIterate(p)
        res = solve!(β, f, g, solver)
    end
    Random.seed!(1)
    t = @elapsed solve!(SparseIterate(p), f, g, solver)
    @printf "%-28s %5d passes   %6.1f ms   objective %.12f\n" name res.passes 1000t res.minimum
    (name = name, β = Vector(β), res = res)
end;

# Here each acceleration takes less than half the passes of its plain order.
# The cost of a pass differs a little: a pass with momentum also moves the
# cached residuals with the extrapolated point, and Anderson extrapolation
# solves a K × K linear system every K + 1 passes. Which order wins depends on
# the design; the random order, the default, is the safer choice when many
# columns are strongly correlated with each other.

@assert all(r -> r.res.converged, results)
@assert maximum(r -> abs(r.res.minimum - results[1].res.minimum), results) < 1e-8 * results[1].res.minimum
@assert maximum(r -> norm(r.β - results[1].β, Inf), results) < 1e-4
@assert results[2].res.passes < results[1].res.passes
@assert results[4].res.passes < results[3].res.passes
