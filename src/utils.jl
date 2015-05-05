###################################
# tracing based on optim package
# removed grad norm field
#

immutable OptimizationState
    iteration::Int
    value::Float64
    metadata::Dict
end

function OptimizationState(i::Integer, f::Real)
    OptimizationState(int(i), @compat(Float64(f)), Dict())
end

immutable OptimizationTrace
    states::Vector{OptimizationState}
end

OptimizationTrace() = OptimizationTrace(Array(OptimizationState, 0))

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   \n" t.iteration t.value
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, t::OptimizationTrace)
    @printf io "Iter     Function value \n"
    @printf io "------   -------------- \n"
    for state in t.states
        show(io, state)
    end
    return
end


function update!(tr::OptimizationTrace,
                 iteration::Int64,
                 f_x::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool,
                 printEvery::Int64)
    os = OptimizationState(iteration, f_x, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace && mod(iteration, printEvery) == 0
        show(os)
    end
    return
end

macro gdtrace()
  quote
    if tracing
      dt = Dict()
      if extended_trace
        dt["x"] = copy(x)
        dt["lambda"] = lambda
      end
      update!(tr,
              iter,
              curVal,
              dt,
              store_trace,
              show_trace,
              printEvery)
    end
  end
end

###################################

type ProximalOptions
  maxiter::Int64
  ftol::Float64
  xtol::Float64
  store_trace::Bool
  show_trace::Bool
  extended_trace::Bool
  printEvery::Int64
end

function ProximalOptions(;maxiter::Integer     = 200,
                         ftol::Real            = 1.0e-6,
                         xtol::Real            = 1.0e-8,
                         store_trace::Bool     = false,
                         show_trace::Bool      = false,
                         extended_trace::Bool  = false,
                         printEvery::Integer   = 1)

  maxiter > 1 || error("maxiter must be an integer greater than 1.")
  ftol > 0 || error("ftol must be a positive real value.")
  xtol > 0 || error("xtol must be a positive real value.")

  ProximalOptions(convert(Int64, maxiter),
                  convert(Float64, ftol),
                  convert(Float64, xtol),
                  store_trace,
                  show_trace,
                  extended_trace,
                  convert(Int64, printEvery)
                  )
end

function check_optim_done{T<:FloatingPoint}(iter::Integer,
                                            curval::T, lastval::T,
                                            x::StridedArray{T}, z::StridedArray{T},
                                            options::ProximalOptions)
  iter >= options.maxiter || abs(curval-lastval) < convert(T, options.ftol) || _l2diff(z, x) < convert(T, options.xtol)
end


function _sqrl2diff{T<:FloatingPoint}(x::StridedArray{T}, y::StridedArray{T})
    @assert length(x) == length(y)
    s = zero(T)
    @inbounds for i = 1:length(x)
        s += abs2(x[i] - y[i])
    end
    return s
end

_l2diff{T<:FloatingPoint}(x::StridedArray{T}, y::StridedArray{T}) =
    sqrt(_sqrl2diff(x, y))

