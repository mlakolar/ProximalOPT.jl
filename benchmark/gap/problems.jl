# The lasso designs A–E of benchmark/lasso_bench.jl, for
#
#     F(β) = ‖y - Xβ‖² / (2n) + λ‖β‖₁,
#
# given to the solvers as LeastSquaresLoss(y/√n, X/√n).

using LinearAlgebra, Random, Printf
using ProximalBase, ProximalSolvers

const DESIGNS = [
  (name = "A", n = 500, p = 2500, s = 50, ρ = 0.0, common = 0.0),
  (name = "B", n = 500, p = 2500, s = 50, ρ = 0.8, common = 0.0),
  (name = "C", n = 5000, p = 500, s = 50, ρ = 0.5, common = 0.0),
  (name = "D", n = 1000, p = 20000, s = 20, ρ = 0.0, common = 0.0),
  (name = "E", n = 500, p = 2500, s = 50, ρ = 0.0, common = 0.5),
]

# Gaussian columns with AR(1) correlation ρ or a common factor, scaled to
# ‖Xⱼ‖² = n, s coefficients ±1, signal-to-noise ratio 4 (as lasso_setup.jl)
function design(d; seed = 1, snr = 4.0)
  n, p, s = d.n, d.p, d.s
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if d.ρ > 0
    c = sqrt(1 - d.ρ^2)
    for j in 2:p
      @views X[:, j] .= d.ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  d.common > 0 && (X .= sqrt(1 - d.common) .* X .+ sqrt(d.common) .* randn(rng, n))
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  y = signal .+ sqrt(sum(abs2, signal) / n / snr) .* randn(rng, n)
  f = LeastSquaresLoss(y ./ sqrt(n), X ./ sqrt(n))
  λmax = lambda_max(f, NormL1(1.0), SparseIterate(p))
  f, λmax, p
end

# the fastest of `rounds` runs of run() (which returns the result), with the result
function fastest(run; rounds = 3)
  best, res = Inf, nothing
  for _ in 1:rounds
    GC.gc()
    t = @elapsed r = run()
    t < best && ((best, res) = (t, r))
  end
  best, res
end
