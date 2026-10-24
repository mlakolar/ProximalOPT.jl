# Instantiate both environments before running under a Slurm allocation.
using Pkg
Pkg.activate(@__DIR__)
if haskey(ENV, "PROXIMAL_BASE_PATH")
  Pkg.develop([PackageSpec(path = ENV["PROXIMAL_BASE_PATH"]),
               PackageSpec(path = dirname(@__DIR__))])
elseif VERSION < v"1.11"
  # Julia 1.10 ignores [sources]. The development losses are not all registered.
  Pkg.develop([PackageSpec(url = "https://github.com/mlakolar/ProximalBase.jl"),
               PackageSpec(path = dirname(@__DIR__))])
end
# A manifest from another Julia version can retain incompatible stdlib entries.
# Resolve for the active runtime before instantiating either environment.
Pkg.resolve()
Pkg.instantiate()
Pkg.activate(joinpath(@__DIR__, "legacy_cd"))
Pkg.resolve()
Pkg.instantiate()
