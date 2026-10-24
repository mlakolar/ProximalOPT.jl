# The table of the benchmark: per problem, mixture and package, the median time
# of the path and its objective values against the best of all packages at each
# level, the objective computed here with dense algebra.
#
#   julia --project=benchmark/sgl benchmark/sgl/summary.jl PROBROOT RESULTROOT

using LinearAlgebra, Printf

include(joinpath(@__DIR__, "common.jl"))

objective(X, y, groups, α, λ, b) =
  sum(abs2, y - X * b) / (2 * size(X, 1)) + λ * (α * norm(b, 1) + (1 - α) * sum(sqrt(length(G)) * norm(b[G]) for G in groups))

const PACKAGES = ("ProximalSolvers", "ProximalSolvers-gapsafe", "SGL", "SGL-tight", "sparsegl")

function main()
  proot, rroot = ARGS[1], ARGS[2]
  println("| problem | α | package | median time (s) | max rel. excess objective | levels worse than 1e-6 |")
  println("|---|---|---|---|---|---|")
  for name in sort(readdir(proot))
    isdir(joinpath(proot, name)) || continue
    X, y, groups = readproblem(joinpath(proot, name))
    n, p = size(X)
    for αs in ("0.05", "0.5", "0.95")
      α = parse(Float64, αs)
      lf = joinpath(proot, name, "lambda_$(αs).bin")
      isfile(lf) || continue
      λs = Vector(reinterpret(Float64, read(lf)))
      res = joinpath(rroot, name, αs)
      F = Dict{String, Vector{Float64}}()
      T = Dict{String, Float64}()
      for pkg in PACKAGES
        bf = joinpath(res, "beta_$(pkg).bin")
        isfile(bf) || continue
        B = reshape(Vector(reinterpret(Float64, read(bf))), p, length(λs))
        F[pkg] = [objective(X, y, groups, α, λs[k], B[:, k]) for k in eachindex(λs)]
        T[pkg] = (ts = parse.(Float64, split(readline(joinpath(res, "time_$(pkg).txt")))); sort(ts)[(length(ts) + 1) ÷ 2])
      end
      isempty(F) && continue
      best = [minimum(F[k][i] for k in keys(F)) for i in eachindex(λs)]
      for pkg in PACKAGES
        haskey(F, pkg) || continue
        rel = (F[pkg] .- best) ./ best
        @printf "| %s (n=%d, p=%d) | %s | %s | %.3f | %.1e | %d |\n" name n p αs pkg T[pkg] maximum(rel) count(>(1e-6), rel)
      end
    end
  end
end

main()
