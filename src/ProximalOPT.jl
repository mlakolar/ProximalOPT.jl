module ProximalOPT

# using Devectorize
import Optim.update!

export
  # types
  DifferentiableFunction,
  ProximalOperator,
  # proximal minimization algorithms
  prox_grad!,
  acc_prox_grad!,
  # smooth functions
  quad_f,
  # proximal operators
  prox_l1,
  prox_l2


# Types
include("types.jl")

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
function prox_grad!(x::Vector{Float64}, fmin::DifferentiableFunction, prox_op::ProximalOperator;
                    beta::Float64         = 0.8,
                    MAX_ITER::Int64       = 300,
                    ABSTOL::Float64       = 1e-4,
                    MIN_LAMBDA::Float64   = 1e-10,
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

  fx::Float64 = fmin.fg!(tmp_x, x)
  gx::Float64 = prox_op.g(x)
  curVal::Float64 = fx + gx
  lastVal = curVal
  fz::Float64 = 0.
  norm_z::Float64 = 0.

  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace

  for iter = 1:MAX_ITER
    fmin.g!(grad_x, x)
    while true
      for i=1:length(x)
        tmp_x[i] = x[i] - lambda * grad_x[i]
      end
      prox_op.prox_g!(z, tmp_x, lambda)
      fz = fmin.fg!(tmp_x, z)
      for i=1:length(x)
        tmp_x[i] = z[i] - x[i]
      end
      sm_square = norm(tmp_x)^2.
      if fz <= fx + dot(grad_x, tmp_x) + sm_square / 2. / lambda
        break
      end
      lambda = beta*lambda;
      if lambda < MIN_LAMBDA
        break
      end
    end
    copy!(x, z)
    fx = fz
    gx = prox_op.g(x)
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
function acc_prox_grad!(x::Vector{Float64}, fmin::DifferentiableFunction, prox_op::ProximalOperator;
                        beta::Float64     = 0.8,
                        MAX_ITER::Int64   = 300,
                        ABSTOL::Float64   = 1e-4,
                        MIN_LAMBDA::Float64   = 1e-10,
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
  grad_y = copy(x)
  z = copy(x)
  prev_x = copy(x)

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
    fy = fmin.fg!(tmp, y)
    fmin.g!(grad_y, y)
    while true
      for k=1:length(x)
        tmp[k] = y[k] - lambda * grad_y[k]
      end
      prox_op.prox_g!(z, tmp, lambda)
      fz = fmin.fg!(tmp, z)
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

    gx = prox_op.g(x)
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
