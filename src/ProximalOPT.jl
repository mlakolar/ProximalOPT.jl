module ProximalOPT

using Compat
using Devectorize

import Optim.OptimizationState
import Optim.OptimizationTrace
import Optim.update!

export
  # types
  DifferentiableFunction,
  ProximableFunction,

  # proximal minimization algorithms
  prox_grad!,
  acc_prox_grad!,

  # smooth functions
  QuadraticFunction,
  gradient!, value_and_gradient!,

  # proximal functions
  ProxL1, ProxL2, ProxL2Sq, ProxNuclear, ProxSumProx,
  ProxL1L2, ProxL1Nuclear,
  value, prox!

# Helper functions to construct DifferentiableFunctions
include("diff_functions.jl")

# Helper functions to construct ProximalOperator
include("proximal_functions.jl")

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
              NaN,
              dt,
              store_trace,
              show_trace)
    end
  end
end


# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function prox_grad!{T<:FloatingPoint}(x::Array{T}, fmin::DifferentiableFunction, prox_op::ProximableFunction;
                    beta::T               = 0.8,
                    MAX_ITER::Int64       = 300,
                    ABSTOL::T             = 1e-4,
                    MIN_LAMBDA::T         = 1e-10,
                    store_trace::Bool     = false,
                    show_trace::Bool      = false,
                    extended_trace::Bool  = false
                    )

  if extended_trace
    store_trace = true
  end
  if show_trace
    @printf "Iter     Function value   Gradient norm \n"
  end

  iter::Int64 = 0
  lastVal::Float64 = 0.
  lambda::Float64 = 1.

  tmp_x = copy(x)
  grad_x = copy(x)
  z = copy(x)

  fx::Float64 = value_and_gradient!(fmin, tmp_x, x)
  gx::Float64 = value(prox_op, x)
  curVal::Float64 = fx + gx
  lastVal = curVal
  fz::Float64 = 0.
  norm_z::Float64 = 0.

  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace

  for iter = 1:MAX_ITER
    gradient!(fmin, grad_x, x)
    while true
      @devec tmp_x = x - lambda .* grad_x
      prox!(prox_op, tmp_x, x, lambda)
      fz = value_and_gradient!(fmin, tmp_x, z)
      @devec tmp_x = z - x
      if fz <= fx + dot(grad_x, tmp_x) + sumabs2(tmp_x) / 2. / lambda
        break
      end
      lambda = beta*lambda
      if lambda < MIN_LAMBDA
        break
      end
    end
    x, z = z, x
    fx = fz
    value(prox_op, x)
    curVal = fx + gx
    @gdtrace
    if abs(curVal - lastVal) < ABSTOL
        break
    end
    lastVal = curVal
  end

  tr
end


# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function acc_prox_grad!{T<:FloatingPoint}(x::Array{T}, fmin::DifferentiableFunction, prox_op::ProximableFunction;
                        beta::T           = 0.8,
                        MAX_ITER::Int64   = 300,
                        ABSTOL::T         = 1e-4,
                        MIN_LAMBDA::T     = 1e-10,
                        store_trace::Bool = false,
                        show_trace::Bool = false,
                        extended_trace::Bool = false,
                        )

  if extended_trace
    store_trace = true
  end
  if show_trace
    @printf "Iter     Function value   Gradient norm \n"
  end

  iter = 0
  lambda = 1.

  y = copy(x)

  tmp = copy(x)
  tmp_v = view(tmp)
  grad_y = copy(x)
  grad_y_v = view(grad_y)
  z = copy(x)
  z_v = view(z)
  prev_x = copy(x)

  x_v = view(x)
  y_v = view(y)

  fy::Float64 = 0.
  fz::Float64 = 0.
  gx::Float64 = 0.

  curVal::Float64 = Inf
  lastVal::Float64 = curVal

  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace

  for iter = 1:MAX_ITER
    omega = float64(iter / (iter+3.))
    for k=1:length(x)
      y[k] = x[k] +  omega * (x[k] - prev_x[k])
    end
    fy = fmin.fg!(tmp_v, y_v)
    fmin.fg!(grad_y_v, y_v)
    while true
      for k=1:length(x)
        tmp[k] = y[k] - lambda * grad_y[k]
      end
      prox_op.prox_g!(z_v, tmp_v, lambda)
      fz = fmin.fg!(tmp_v, z_v)
      for k=1:length(x)
        tmp[k] = z[k] - x[k]
      end
      sm_square = norm(tmp)^2.
      if fz <= fy + dot(grad_y, tmp) + sm_square / 2. / lambda
        break
      end
      lambda = beta * lambda
      if lambda < MIN_LAMBDA
        break
      end
    end
    copy!(prev_x, x)
    copy!(x, z)

    gx = prox_op.g(x_v)
    curVal = fz + gx
    @gdtrace
    if abs(curVal - lastVal) < ABSTOL
        break
    end
    lastVal = curVal
  end

  tr
end

end
