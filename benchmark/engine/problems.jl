# Problems shared by the runners of the old and the new code (old/run.jl and
# new/run.jl). This file uses only standard libraries, so that both
# environments load it and draw the same data: the data depend only on the
# seeds and on the Julia version, which must be the same for both runs.

using LinearAlgebra, Random, Statistics

################################################################################
# lasso designs, as in benchmark/lasso_setup.jl

"""
Gaussian design with AR(1) correlation ρ between neighboring columns, or with
correlation `common` between every pair of columns, columns scaled so that
‖Xⱼ‖² = n, s nonzero coefficients ±1 at random positions, and noise with
signal-to-noise ratio `snr`.
"""
function lasso_design(; n, p, s, ρ = 0.0, common = 0.0, snr = 4.0, seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if ρ > 0
    c = sqrt(1 - ρ^2)
    for j in 2:p
      @views X[:, j] .= ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  common > 0 && (X .= sqrt(1 - common) .* X .+ sqrt(common) .* randn(rng, n))
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  σ = sqrt(sum(abs2, signal) / n / snr)
  (X = X, y = signal .+ σ .* randn(rng, n), β = β)
end

const LASSO_DESIGNS = (
  A = (n = 500,  p = 2500,  s = 50),
  B = (n = 500,  p = 2500,  s = 50, ρ = 0.8),
  C = (n = 5000, p = 500,   s = 50, ρ = 0.5),
  D = (n = 1000, p = 20000, s = 20),
  E = (n = 500,  p = 2500,  s = 50, common = 0.5),
)

const _DESIGN_CACHE = Dict{Symbol, Any}()
design(name::Symbol) = get!(() -> lasso_design(; LASSO_DESIGNS[name]...), _DESIGN_CACHE, name)

# the smallest λ with a zero lasso solution, ‖X'y‖∞ / n, without an intercept
lasso_λmax(X, y) = maximum(abs, X' * y) / size(X, 1)

lasso_objective(X, y, β, λ) = sum(abs2, y - X * β) / (2 * size(X, 1)) + λ * sum(abs, β)

# groups of `size` consecutive columns
consecutive_groups(p, size) = [i:min(i + size - 1, p) for i in 1:size:p]

# the smallest λ with a zero group lasso solution, maxₖ ‖X[:, Gₖ]'y‖ / n
group_λmax(X, y, groups) = (g = X' * y ./ size(X, 1); maximum(G -> norm(g[G]), groups))

group_objective(X, y, β, λ, groups) =
  sum(abs2, y - X * β) / (2 * size(X, 1)) + λ * sum(G -> norm(β[G]), groups)

# the column standard deviations sqrt(∑ᵢ Xᵢⱼ² / n), without centering
column_scales(X) = vec(sqrt.(sum(abs2, X; dims = 1) ./ size(X, 1)))

################################################################################
# varying coefficients, as in CoordinateDescent's benchmark/locpoly_bench.jl

# varying coefficients βⱼ(z) = sin(bⱼ z), with bⱼ drawn from {2, 4, 6, 8}
function varying_coefficient_data(n, p; seed = 1)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  Z = rand(rng, n)
  ε = 0.1 .* randn(rng, n)
  b = [rand(rng, (2, 4, 6, 8)) for j in 1:p]
  Y = [dot(sin.(b .* Z[i]), X[i, :]) for i in 1:n] .+ ε
  (X = X, Y = Y, Z = Z)
end

################################################################################
# precision matrices, as in CovSel's benchmark/benchmarks.jl

"""
Sample covariances of n observations from two groups, with Σx = I and Σy the
AR(1) covariance with ρ = 0.7; the difference of the precision matrices,
I - inv(Σy), is tridiagonal. Returns the covariances and the samples.
"""
function difference_problem(p; n = 1000, seed = p)
  rng = Xoshiro(seed)
  Σy = [0.7^abs(i - j) for i in 1:p, j in 1:p]
  X = randn(rng, n, p)
  Y = randn(rng, n, p) * cholesky(Σy).U
  (Sx = Symmetric(cov(X)), Sy = Symmetric(cov(Y)), X = X, Y = Y)
end

# the direct difference loss tr(Sx Δ Sy Δ)/2 - tr(Δ (Sy - Sx)) plus λ times the
# entrywise ℓ1 norm of Δ, both diagonal and off-diagonal
dtrace_objective(Sx, Sy, Δ, λ) = tr(Sx * Δ * Sy * Δ) / 2 - tr(Δ * (Sy - Sx)) + λ * sum(abs, Δ)

# the smallest λ with a zero direct difference solution, ‖Sy - Sx‖∞ entrywise
dtrace_λmax(Sx, Sy) = maximum(abs, Sy - Sx)
