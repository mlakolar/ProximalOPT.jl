# The sparse-group-lasso problems of the benchmark, written as raw Float64 files
# that R reads with readBin: X (column-major), y, the group of each column, and
# per mixture α the grid of levels from ProximalSolvers' exact λmax.
#
#   julia --project=benchmark/sgl benchmark/sgl/problems.jl OUTDIR [SMOKE]

using ProximalBase, ProximalSolvers, Random, LinearAlgebra, Printf

const PROBLEMS = (
  (name = "A", n = 200, p = 1000, gsize = 10, active = 5, seed = 1),
  (name = "B", n = 500, p = 5000, gsize = 20, active = 8, seed = 2),
)
const SMOKE_PROBLEMS = ((name = "S", n = 50, p = 120, gsize = 6, active = 3, seed = 3),)
const ALPHAS = (0.05, 0.5, 0.95)
const NLAMBDA = 50
const RATIO = 0.01

groups_of(p, gsize) = [i:i+gsize-1 for i in 1:gsize:p]

# columns correlated 0.3 within a group, half the entries of the active groups
# nonzero, noise for a signal-to-noise ratio of about 2, all centered
function problem(pr)
  rng = Xoshiro(pr.seed)
  groups = groups_of(pr.p, pr.gsize)
  X = randn(rng, pr.n, pr.p)
  ρ = 0.3
  for G in groups
    z = randn(rng, pr.n)
    X[:, G] .= sqrt(1 - ρ) .* X[:, G] .+ sqrt(ρ) .* z
  end
  β = zeros(pr.p)
  for k in 1:pr.active
    G = groups[k]
    for j in G[1:2:end]
      β[j] = (0.5 + rand(rng)) * rand(rng, (-1, 1))
    end
  end
  μ = X * β
  y = μ .+ (std_(μ) / sqrt(2)) .* randn(rng, pr.n)
  # SGL always centers X and y (and fits an intercept); centered data make the
  # problems without intercept of the three packages the same
  X .-= sum(X; dims = 1) ./ pr.n
  y .-= sum(y) / pr.n
  X, y, groups
end
std_(v) = sqrt(sum(abs2, v .- sum(v) / length(v)) / (length(v) - 1))

sgl(α, groups, λ = 1.0) = Scaled(SparseGroupNormL2(α, 1 - α, groups), λ)

if abspath(PROGRAM_FILE) == @__FILE__
  out = ARGS[1]
  probs = length(ARGS) > 1 && ARGS[2] == "SMOKE" ? SMOKE_PROBLEMS : PROBLEMS
  mkpath(out)
  for pr in probs
    X, y, groups = problem(pr)
    d = joinpath(out, pr.name)
    mkpath(d)
    write(joinpath(d, "X.bin"), X)
    write(joinpath(d, "y.bin"), y)
    write(joinpath(d, "group.bin"), Float64[k for (k, G) in enumerate(groups) for _ in G])
    open(joinpath(d, "meta.txt"), "w") do io
      println(io, "n=$(pr.n) p=$(pr.p) gsize=$(pr.gsize) active=$(pr.active) seed=$(pr.seed)")
    end
    f = (1 / pr.n) * LeastSquaresLoss(y, X)
    for α in ALPHAS
      λmax = lambda_max(f, sgl(α, groups), AtomIterate(pr.p, groups))
      λs = λmax .* RATIO .^ range(0, 1; length = NLAMBDA)
      write(joinpath(d, "lambda_$(α).bin"), collect(λs))
      @printf "%s n=%d p=%d α=%.2f λmax=%.6g\n" pr.name pr.n pr.p α λmax
    end
  end
end
