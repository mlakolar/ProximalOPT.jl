# Run within a Discovery allocation with this checkout in the active environment.
using LinearAlgebra, ProximalBase, ProximalSolvers, Printf
BLAS.set_num_threads(1)

function median_measure(run, prepare; samples = 21)
  prepare(); run()
  prepare(); bytes = @allocated run()
  times = Float64[]
  for _ in 1:samples
    prepare()
    push!(times, @elapsed run())
  end
  sort!(times)
  (ns = 1e9 * times[(samples + 1) ÷ 2], bytes = bytes)
end

function main()
  p = 100_000
  f, g = SqrNormL2(), NormL1(0.1)
  options = ProximalOptions(; maxiter = 2)
  println("solver,fresh_ns,workspace_ns,fresh_bytes,workspace_bytes")
  for method in (ProximalGradientDescent(), AccProxGradDescent(), MonotoneAccProxGradDescent(), ADMMSolver())
    fresh_x = ones(p)
    start = ones(p)
    ws = SolverWorkspace(copy(start), f, g, method, options)
    a = median_measure(() -> solve!(fresh_x, f, g, method, options), () -> copyto!(fresh_x, start))
    b = median_measure(() -> solve!(ws), () -> reinit!(ws, start))
    @printf "%s,%.0f,%.0f,%d,%d\n" nameof(typeof(method)) a.ns b.ns a.bytes b.bytes
  end
end
main()
