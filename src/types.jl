abstract type ProximalSolver end


struct ProximalOptions{TCallback <: Union{Nothing, Function}}
  maxiter::Int64
  ftol::Float64
  xtol::Float64
  gradtol::Float64
  f_calls_limit::Int
  g_calls_limit::Int
  allow_f_increases::Bool
  store_trace::Bool
  show_trace::Bool
  extended_trace::Bool
  printEvery::Int64
  callback::TCallback
  time_limit::Float64
end

function ProximalOptions(;maxiter::Integer       = 200,
                         ftol::Real              = 1.0e-6,
                         xtol::Real              = 1.0e-8,
                         gradtol::Real           = 1.0e-5,
                         f_calls_limit::Int      = 0,
                         g_calls_limit::Int      = 0,
                         allow_f_increases::Bool = false,
                         store_trace::Bool       = false,
                         show_trace::Bool        = false,
                         extended_trace::Bool    = false,
                         printEvery::Integer     = 1,
                         callback                = nothing,
                         time_limit              = Inf)

  maxiter > 1 || error("maxiter must be an integer greater than 1.")
  ftol > 0 || error("ftol must be a positive real value.")
  xtol > 0 || error("xtol must be a positive real value.")
  printEvery = printEvery > 0 ? printEvery : 1

  ProximalOptions{typeof(callback)}(
                  convert(Int64, maxiter),
                  convert(Float64, ftol),
                  convert(Float64, xtol),
                  convert(Float64, gradtol),
                  f_calls_limit,
                  g_calls_limit,
                  allow_f_increases,
                  store_trace,
                  show_trace,
                  extended_trace,
                  convert(Int64, printEvery),
                  callback,
                  time_limit
                  )
end

function print_header(options::ProximalOptions)
  if options.show_trace
    @printf "Iter     Function value       |dx|/|x|       Step Size\n"
  end
end

function print_header(method::ProximalSolver, options::ProximalOptions)
    @printf "Iter     Function value       |dx|/|x|       Step Size\n"
end


struct OptimizationState{T <: ProximalSolver}
    iteration::Int
    fg_x::Float64
    dxx::Float64
    L::Float64
    metadata::Dict
end

OptimizationTrace{T} = Vector{OptimizationState{T}}


function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e   %14e\n" t.iteration t.fg_x t.dxx 1. / t.L
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   |dx|/|x|       Step Size\n"
    @printf io "------   --------------   --------       ---------\n"
    for state in tr
        show(io, state)
    end
    return
end



mutable struct OptimizationResults{O<:ProximalSolver,T,N,M}
    method::O
    minimizer::Array{T,N}
    minimum::T
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::Float64
    x_residual::Float64
    f_converged::Bool
    f_tol::Float64
    f_residual::Float64
    grad_converged::Bool
    g_tol::Float64
    g_residual::Float64
    f_increased::Bool
    trace::OptimizationTrace{M}
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

Base.summary(r::OptimizationResults) = summary(r.method)
minimizer(r::OptimizationResults) = r.minimizer
minimum(r::OptimizationResults) = r.minimum
iterations(r::OptimizationResults) = r.iterations
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged
converged(r::OptimizationResults) = r.x_converged || r.f_converged || r.grad_converged
x_converged(r::OptimizationResults) = r.x_converged
f_converged(r::OptimizationResults) = r.f_converged
f_increased(r::OptimizationResults) = r.f_increased
grad_converged(r::OptimizationResults) = r.grad_converged

x_tol(r::OptimizationResults) = r.x_tol
x_residual(r::OptimizationResults) = r.x_residual
f_tol(r::OptimizationResults) = r.f_tol
f_residual(r::OptimizationResults) = r.f_residual
g_tol(r::OptimizationResults) = r.g_tol
g_residual(r::OptimizationResults) = r.g_residual


function Base.show(io::IO, r::OptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    # if length(join(initial_state(r), ",")) < 40
    #     @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    # else
    #     @printf io " * Starting Point: [%s, ...]\n" join(initial_state(r)[1:2], ",")
    # end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(minimizer(r)[1:2], ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s \n" x_tol(r) x_converged(r)
    @printf io "     |x - x'| = %.2e \n"  x_residual(r)
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" f_tol(r) f_converged(r)
    @printf io "     |f(x) - f(x')| / |f(x)| = %.2e \n" f_residual(r)
    @printf io "   * |g(x)| < %.1e: %s \n" g_tol(r) grad_converged(r)
    @printf io "     |g(x)| = %.2e \n"  g_residual(r)
    @printf io "   * stopped by an increasing objective: %s\n" f_increased(r)
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    # @printf io " * Objective Calls: %d" f_calls(r)
    # @printf io "\n * Gradient Calls: %d" g_calls(r)
    return
end
