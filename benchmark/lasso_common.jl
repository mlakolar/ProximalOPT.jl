using LinearAlgebra, Random, Statistics, SparseArrays, Printf

struct LassoProblem
  name::String
  desc::String
  X::Matrix{Float64}
  y::Vector{Float64}
  Xs::Matrix{Float64}                  # X / √n
  ys::Vector{Float64}                  # y / √n
  n::Int
  p::Int
  λmax::Float64                        # smallest λ whose solution is zero
  G::Union{Nothing, Matrix{Float64}}   # X'X / n
  mc::Vector{Float64}                  # -X'y / n
  gram_time::Float64                   # seconds to form G
end

"""
Gaussian design with AR(1) correlation ρ between neighboring columns, or with
correlation `common` between every pair of columns (a common factor), columns
scaled so that ‖Xⱼ‖² = n, s nonzero coefficients ±1 at random positions, and
noise with signal-to-noise ratio `snr`. X'X/n is formed when `gram` is true.
"""
function make_problem(name, desc; n, p, s, ρ = 0.0, common = 0.0, snr = 4.0, seed = 1, gram = true)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  if ρ > 0
    c = sqrt(1 - ρ^2)
    for j in 2:p
      @views X[:, j] .= ρ .* X[:, j-1] .+ c .* X[:, j]
    end
  end
  if common > 0
    X .= sqrt(1 - common) .* X .+ sqrt(common) .* randn(rng, n)
  end
  for j in 1:p
    @views X[:, j] ./= sqrt(sum(abs2, X[:, j]) / n)
  end
  β = zeros(p)
  β[randperm(rng, p)[1:s]] .= rand(rng, (-1.0, 1.0), s)
  signal = X * β
  σ = sqrt(sum(abs2, signal) / n / snr)
  y = signal .+ σ .* randn(rng, n)

  mc = -(X' * y) ./ n
  G, gram_time = nothing, NaN
  if gram
    gram_time = @elapsed begin
      G = X' * X
      G ./= n
    end
  end
  LassoProblem(name, desc, X, y, X ./ sqrt(n), y ./ sqrt(n), n, p, maximum(abs, mc), G, mc, gram_time)
end

objective(P::LassoProblem, β, λ) = sum(abs2, P.y - P.X * β) / (2 * P.n) + λ * sum(abs, β)

# largest violation of the optimality conditions ∇f(β) + λ ∂‖β‖₁ ∋ 0
function lasso_kkt(P::LassoProblem, β, λ)
  g = P.X' * (P.X * β - P.y) ./ P.n
  r = 0.0
  for j in eachindex(β)
    r = max(r, β[j] == 0 ? max(abs(g[j]) - λ, 0.0) : abs(g[j] + λ * sign(β[j])))
  end
  r
end

# relative suboptimality (F - F*) / F*
relsubopt(F, Fstar) = max((F - Fstar) / abs(Fstar), 0.0)

# coordinates whose zero pattern differs from the reference
support_errors(β, βstar) = count(j -> iszero(β[j]) != iszero(βstar[j]), eachindex(β))

"""
Run `f()` at least `minreps` times, and more until the total time reaches
`budget` seconds (at most `maxreps` runs). Returns the first output and the
minimum time: other load on the machine only ever adds time.
"""
function timed(f; budget = 1.0, minreps = 3, maxreps = 7)
  t = @elapsed out = f()
  times = [t]
  while length(times) < minreps || (sum(times) < budget && length(times) < maxreps)
    push!(times, @elapsed f())
  end
  out, minimum(times)
end

env_summary() = string("Julia ", VERSION, ", ", Sys.cpu_info()[1].model, ", BLAS threads = ",
                       BLAS.get_num_threads(), ", ", basename(BLAS.get_config().loaded_libs[1].libname))
