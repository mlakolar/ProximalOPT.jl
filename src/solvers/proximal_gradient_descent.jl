



# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function solve!{T<:AbstractFloat}(
    ::ProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::Real = 0.5,
    options::ProximalOptions=ProximalOptions()
    )

  store_trace = options.store_trace
  show_trace = options.show_trace
  extended_trace = options.extended_trace
  printEvery = options.printEvery

  if extended_trace
    store_trace = true
  end
  if show_trace
    @printf "Iter     Function value   Gradient norm \n"
  end

  iter = 0
  lambda = one(T)

  tmp_x = similar(x)
  grad_x = similar(x)
  z = similar(x)

  fx = value_and_gradient!(f, grad_x, x)
  gx = value(g, x)::T
  curVal = fx + gx

  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
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
