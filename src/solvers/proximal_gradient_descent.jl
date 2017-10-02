struct ProximalGradientDescent{L} <: ProximalSolver
  linesearch!::L
end

Base.summary(::ProximalGradientDescent) = "Proximal Gradient Descent"

ProximalGradientDescent(; linesearch = LineSearches.BackTracking()) =
  ProximalGradientDescent{typeof(linesearch)}(linesearch)

mutable struct ProximalGradientDescentState{T,N}
    x::Array{T,N}
    f_x::T
    g_x::T
    grad_f_x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    g_x_previous::T
    grad_f_x_previous::Array{T,N}
    @add_linesearch_fields()
end


function initial_state(method::ProximalGradientDescent, options, f, g, x0::Array{T}) where T

    grad_f_x = similar(x0)
    f_x = value_and_gradient!(f, grad_f_x, x0)
    g_x = value(g, x0)

    GradientDescentState(x0,          # Maintain current state in state.x
                         f_x,         # Maintain current f(x) in state.f_x
                         g_x,         # Maintain current g(x) in state.g_x
                         grad_f_x,    # Maintain gradient of f at current x in state.grad_f_x
                         similar(x0), # Maintain previous state in state.x_previous
                         T(NaN),      # Store previous f in state.f_x_previous
                         T(NaN),      # Store previous g in state.g_x_previous
                         similar(x0), # Maintain previous gradient of f in state.grad_f_x_previous for convergence check
                         @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function trace!(tr, f, g, state, iteration, method::ProximalGradientDescent, options)
  common_trace!(tr, f, g, state, iteration, method, options)
end


# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function solve!{T<:AbstractFloat}(
    ::ProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::Real = 0.5,
    options::ProximalOptions=ProximalOptions()
    )


  iter = 0
  lambda = one(T)

  tmp_x = similar(x)
  grad_x = similar(x)
  z = similar(x)

  fx = value_and_gradient!(f, grad_x, x)
  gx = value(g, x)::T
  curVal = fx + gx

  while true
    iter += 1

    lastVal = curVal
    while true
      _y_minus_ax!(tmp_x, x, lambda, grad_x)
      prox!(g, z, tmp_x, lambda)
      fz = value(f, z)
      if fz <= _taylor_value(fx, z, x, grad_x, lambda)
        break
      end
      lambda = beta*lambda
      if lambda < eps(T)
        break
      end
    end
    x, z = z, x
    fx = value_and_gradient!(f, grad_x, x)::T
    gx = value(g, x)::T
    curVal = fx + gx
    @gdtrace
    if check_optim_done(iter, curVal, lastVal, x, z, options)
      break
    end
    lastVal = curVal
  end
  tr
end
