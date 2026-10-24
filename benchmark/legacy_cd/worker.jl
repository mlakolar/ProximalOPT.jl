using CoordinateDescent, ProximalBase, Serialization
using LinearAlgebra, Random, SparseArrays, Statistics, Printf
include(joinpath(@__DIR__, "..", "lasso_common.jl"))
BLAS.set_num_threads(1)

cd_options(tol; kwargs...) = CDOptions(; optTol = Float64(tol), maxIter = 1_000_000, kwargs...)

function legacy_solve(operation, P, λ, tol = nothing)
  # Random coordinate orders are reproducible, including timing repeats.
  Random.seed!(20261024)
  if operation === :default
    return Vector(lasso(P.X, P.y, λ).x)
  elseif operation === :path
    path = CoordinateDescent.LassoPath(P.X, P.y, λ, cd_options(tol; randomize = true); standardizeX = false)
    return [Vector(b) for b in path.βpath]
  elseif operation === :path_gram || operation === :path_reference
    x = SparseIterate(P.p)
    f = operation === :path_gram ? CDQuadraticLoss(P.G, P.mc) : CDLeastSquaresLoss(P.y, P.X)
    return [(coordinateDescent!(x, f, ProxL1(level), cd_options(tol; randomize = true)); Vector(x)) for level in λ]
  end
  x = SparseIterate(P.p)
  if operation === :gram
    f, options = CDQuadraticLoss(P.G, P.mc), cd_options(tol; randomize = true)
  elseif operation === :cyclic || operation === :random || operation === :continuation
    f = CDLeastSquaresLoss(P.y, P.X)
    options = cd_options(tol; randomize = operation !== :cyclic, warmStart = operation !== :continuation)
  else
    error("Unknown historical operation: $operation")
  end
  coordinateDescent!(x, f, ProxL1(λ), options)
  Vector(x)
end

function main()
  protocol = stdout
  problems = Dict{UInt, LassoProblem}()
  serialize(protocol, (ok = true, environment = "CoordinateDescent $(pkgversion(CoordinateDescent)), ProximalBase $(pkgversion(ProximalBase)); $(env_summary())"))
  flush(protocol)
  while !eof(stdin)
    request = deserialize(stdin)
    request.operation === :stop && break
    response = try
      request.problem === nothing || (problems[request.key] = request.problem)
      P = problems[request.key]
      # Keep solver output out of the binary protocol, including warnings or
      # progress printed by a historical package.
      result, time = redirect_stdout(stderr) do
        run = () -> legacy_solve(request.operation, P, request.args...)
        if request.timing === nothing
          run(), NaN
        else
          run()                    # compilation is outside the measured repeats
          timed(run; request.timing...)
        end
      end
      (ok = true, result = result, time = time)
    catch err
      (ok = false, error = sprint(showerror, err, catch_backtrace()))
    end
    serialize(protocol, response)
    flush(protocol)
  end
end

main()
