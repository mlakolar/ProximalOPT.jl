abstract type ProximalSolver end


type ProximalOptions{TCallback <: Union{Void, Function}}
  maxiter::Int64
  ftol::Float64
  xtol::Float64
  f_calls_limit::Int
  g_calls_limit::Int
  allow_f_increases::Bool
  store_trace::Bool
  show_trace::Bool
  extended_trace::Bool
  callback::TCallback
  printEvery::Int64
  time_limit::Float64
end

function ProximalOptions(;maxiter::Integer       = 200,
                         ftol::Real              = 1.0e-6,
                         xtol::Real              = 1.0e-8,
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

  ProximalOptions{typeof(callback)}(convert(Int64, maxiter),
                  convert(Float64, ftol),
                  convert(Float64, xtol),
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
    @printf "Iter     Function value       |dx|/|x| \n"
  end
end

function print_header(method::ProximalSolver, options::ProximalOptions)
    @printf "Iter     Function value       |dx|/|x| \n"
end


immutable OptimizationState{T <: ProximalSolver}
    iteration::Int
    value::Float64
    metadata::Dict
end

OptimizationTrace{T} = Vector{OptimizationState{T}}


function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in tr
        show(io, state)
    end
    return
end


## TODO
# type for optimization results
# show for optimization results
