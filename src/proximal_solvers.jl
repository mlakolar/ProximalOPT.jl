

abstract ProximalSolver

immutable ProxGradDescent <: ProximalSolver
end

immutable AccProxGradDescent <: ProximalSolver
end

immutable ActiveAccProxGradDescent <: ProximalSolver
end


# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function solve!{T<:FloatingPoint}(
    ::ProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::Real = 0.8,
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

  fx = value_and_gradient!(f, grad_x, x)::T
  gx = value(g, x)::T
  curVal = fx + gx

  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
  while true
    iter += 1

    lastVal = curVal
    while true
      @inbounds for i in eachindex(x)
        tmp_x[i] = x[i] - lambda * grad_x[i]
      end
      prox!(g, z, tmp_x, lambda)
      fz = value_and_gradient!(f, tmp_x, z)::T
      @inbounds for i in eachindex(x)
        tmp_x[i] = z[i] - x[i]
      end
      if fz <= fx + dot(vec(grad_x), vec(tmp_x)) + sumabs2(tmp_x) / 2. / lambda
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

# implements the algorithm in section 4.3 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function solve!{T<:FloatingPoint}(
    ::AccProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::Real = 0.8,
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

  lambdaChange = 0
  lambda = one(T)

  tmp = similar(x)
  grad_y = similar(x)
  y = copy(x)
  z = similar(x)

  fx::T = value_and_gradient!(f, grad_y, x)
  fy::T = fx
  gx::T = value(g, x)
  curVal::T = fx + gx

  iter = 0
  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
  while true
    iter += 1

    lastVal::T = curVal
    while true
      @inbounds for i in eachindex(x)
        tmp[i] = y[i] - lambda * grad_y[i]
      end
      prox!(g, z, tmp, lambda)
      fz::T = value_and_gradient!(f, tmp, z)
      @inbounds for i in eachindex(x)
        tmp[i] = z[i] - y[i]
      end
      if fz <= fy + dot(vec(grad_y), vec(tmp)) + sumabs2(tmp) / 2. / lambda
        break
      end
      lambda = beta*lambda
      lambdaChange = 0
      if lambda < eps(T)
        break
      end
    end
    lambdaChange += 1
    if lambdaChange == 3
      lambda = lambda / beta
      lambdaChange = 0
    end
    ωk = convert(T, iter / (iter + 3.))
    @inbounds for i in eachindex(x)
      y[i] = (1+ωk)*z[i] - ωk*x[i]
    end
    x, z = z, x
    fy = value_and_gradient!(f, grad_y, y)
    fx = value(f, x)
    gx = value(g, x)
    curVal = fx + gx
    @gdtrace
    if check_optim_done(iter, curVal, lastVal, x, z, options)
      break
    end
    lastVal = curVal
  end
  tr
end


# active set implementation of accelerated proximal gradient descent
function solve!{T<:FloatingPoint}(
    ::ActiveAccProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::Real = 0.8,
    options::ProximalOptions=ProximalOptions(),
    maxoutiter::Int64 = 10
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

  # initialize memory
  tmp = similar(x)
  grad_y = similar(x)
  y = copy(x)
  z = similar(x)

  # initial computation
  activeset = active_set(g, x)
  fx::T = value_and_gradient!(f, grad_y, x, activeset)
  fy::T = fx
  gx::T = value(g, x, activeset)
  curVal::T = fx + gx

  oldactiveset = copy(activeset)
  iter = 0
  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
  for outiter=1:maxoutiter

    # minimize over active set
    iter = 0
    lambda = one(T)
    lambdaChange = 0
    while true
      iter += 1

      lastVal::T = curVal
      while true
        @inbounds for i in eachindex(activeset)
          tmp[activeset[i]] = y[activeset[i]] - lambda * grad_y[activeset[i]]
        end
        prox!(g, z, tmp, lambda, activeset)
        fz::T = value_and_gradient!(f, tmp, z, activeset)

        # h = z - y
        # dgh = dot( gradient(y), h )
        # nhsq = norm(h)^2
        dgh = zero(T)
        nhsq = zero(T)
        @inbounds for i in eachindex(activeset)
          tmp[activeset[i]] = z[activeset[i]] - y[activeset[i]]
          dgh += grad_y[activeset[i]] * tmp[activeset[i]]
          nhsq += tmp[activeset[i]].^2
        end
        if fz <= fy + dgh + nhsq / 2. / lambda
          break
        end
        lambda = beta*lambda
        lambdaChange = 0
        if lambda < eps(T)
          break
        end
      end
      lambdaChange += 1
      if lambdaChange == 3
        lambda = lambda / beta
        lambdaChange = 0
      end
      ωk = convert(T, iter / (iter + 3.))
      @inbounds for i in eachindex(activeset)
        y[activeset[i]] = (1+ωk)*z[activeset[i]] - ωk*x[activeset[i]]
      end
      copy!(z, x)
      fy = value_and_gradient!(f, grad_y, y, activeset)
      fx = value(f, x, activeset)
      gx = value(g, x, activeset)
      curVal = fx + gx
      @gdtrace
      if check_optim_done(iter, curVal, lastVal, x, z, options)
        break
      end
      lastVal = curVal
    end

    # figure out what to add to activeset
    add_violator!(activeset, g, f, x)

    if activeset == oldactiveset
      break
    end
    oldactiveset = copy(activeset)


  end
  tr
end


