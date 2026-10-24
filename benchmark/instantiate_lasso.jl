# Instantiate both environments before running under a Slurm allocation.
using Pkg, TOML

# Julia 1.10.8+ selects a minor-version manifest when the file already exists.
# Keep each runtime's locks separate: a shared manifest can otherwise retain a
# Statistics version that the next Julia runtime cannot resolve.
function activate_lasso_environment(path)
  VERSION >= v"1.10.8" || error("Lasso benchmark setup requires Julia 1.10.8 or later.")
  manifest = joinpath(path, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")
  if !isfile(manifest)
    shared = joinpath(path, "Manifest.toml")
    previous = isfile(shared) ? get(TOML.parsefile(shared), "julia_version", nothing) : nothing
    same_runtime = previous !== nothing &&
      (VersionNumber(previous).major, VersionNumber(previous).minor) == (VERSION.major, VERSION.minor)
    if same_runtime
      cp(shared, manifest)
    else
      # Preserve any foreign shared lock; Pkg fills this valid empty manifest
      # from Project.toml on the first setup for the current Julia minor.
      open(manifest, "w") do io
        TOML.print(io, Dict("julia_version" => string(VERSION),
                            "manifest_format" => "2.0", "deps" => Dict{String, Any}()))
      end
    end
  end
  Pkg.activate(path)
end

activate_lasso_environment(@__DIR__)
if haskey(ENV, "PROXIMAL_BASE_PATH") || haskey(ENV, "COORDINATE_DESCENT_PATH") || VERSION < v"1.11"
  # Julia 1.10 ignores [sources]. The development losses are not all registered.
  base = haskey(ENV, "PROXIMAL_BASE_PATH") ? PackageSpec(path = ENV["PROXIMAL_BASE_PATH"]) :
    PackageSpec(url = "https://github.com/mlakolar/ProximalBase.jl")
  cd = haskey(ENV, "COORDINATE_DESCENT_PATH") ? PackageSpec(path = ENV["COORDINATE_DESCENT_PATH"]) :
    PackageSpec(url = "https://github.com/mlakolar/CoordinateDescent.jl")
  Pkg.develop([base, cd,
               PackageSpec(path = dirname(@__DIR__))])
end
# Refresh dependency metadata within this runtime's own lock.
Pkg.resolve()
Pkg.instantiate()
activate_lasso_environment(joinpath(@__DIR__, "legacy_cd"))
Pkg.resolve()
Pkg.instantiate()
