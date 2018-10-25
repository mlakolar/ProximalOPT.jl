




struct AccProxGradDescent <: ProximalSolver
end

struct ActiveAccProxGradDescent <: ProximalSolver
end

struct ADMMSolver <: ProximalSolver
end



# type ADMMOptions
#   ρ::Float64
#   α::Float64
#   maxiter::Int64
#   abstol::Float64
#   reltol::Float64
# end
#
# ADMMOptions(;ρ::Float64=1.,
#            α::Float64=1.,
#            maxiter::Int64=200,
#            abstol::Float64=1e-4,
#            reltol::Float64=1e-2) = ADMMOptions(ρ, α, maxiter, abstol, reltol)
#
#
# #########################################


# minimizes f(x) + g(x)
# both functions need to be proximable
function solve!{T<:AbstractFloat}(
    ::ADMMSolver,
    X::StridedMatrix{T},
    Z::StridedMatrix{T},
    U::StridedMatrix{T},
    f::ProximableFunction,
    g::ProximableFunction;
    options = ADMMOptions()
    )

  maxiter = options.maxiter
  ρ = options.ρ
  α = options.α
  abstol = options.abstol
  reltol = options.reltol

  p = size(X, 1)
  tmpStorage = zeros(T, (p, p))
  Zold = copy(Z)

  for iter=1:maxiter
    # x-update
    # prox_λf(Z - U)
    @simd for i in eachindex(tmpStorage)
      @inbounds tmpStorage[i] = Z[i] - U[i]
    end
    prox!(f, X, tmpStorage, ρ)

    # z-update with relaxation
    copy!(Zold, Z)
    @simd for i in eachindex(tmpStorage)
      @inbounds tmpStorage[i] = α*X[i] + (one(T)-α)*Z[i] + U[i]
    end
    prox!(g, Z, tmpStorage, ρ)

    # u-update
    @simd for i in eachindex(tmpStorage)
      @inbounds U[i] = tmpStorage[i] - Z[i]
    end

    # check convergence
    r_norm = _normdiff(X, Z)
    s_norm = _normdiff(Z, Zold) * sqrt(ρ)
    eps_pri = p*abstol + reltol * max( vecnorm(X), vecnorm(Z) )
    eps_dual = p*abstol + reltol * ρ * vecnorm(U)
    if r_norm < eps_pri && s_norm < eps_dual
      break
    end
  end
  Z
end



# implements the algorithm in section 4.3 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
function solve!{T<:AbstractFloat}(
    ::AccProxGradDescent,
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

  lambda = one(T)

  tmp = similar(x)
  grad_y = similar(x)
  y = copy(x)
  z = similar(x)

  fx::T = value_and_gradient!(f, grad_y, x)
  fy::T = fx
  gx::T = value(g, x)
  curVal::T = fx + gx

  iter = zero(T)
  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
  while true
    iter += 1.

    lastVal::T = curVal
    while true
      _y_minus_ax!(tmp, y, lambda, grad_y)
      prox!(g, z, tmp, lambda)
      fz::T = value(f, z)
      if fz <= _taylor_value(fy, z, y, grad_y, lambda)
        break
      end
      lambda = beta*lambda
      if lambda < eps(T)
        break
      end
    end
    ωk = iter / (iter + 3.)
    _update_y!(y, z, x, ωk)
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
function solve!{T<:AbstractFloat}(
    ::ActiveAccProxGradDescent,
    x::StridedArray{T},
    f::DifferentiableFunction, g::ProximableFunction;
    beta::T = 0.5,
    beta_min::T = 1e-10,
    options::ProximalOptions=ProximalOptions(),
    maxoutiter::Int64 = 200
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
  z = copy(x)

  # initial computation
  activeset = active_set(g, x)
  if activeset.numActive == 0
    add_violator!(activeset, x, g, f, tmp)
  end
  fx::T = value_and_gradient!(f, grad_y, x, activeset)
  fy::T = fx
  gx::T = value(g, x, activeset)
  curVal::T = fx + gx

  iter = 0
  tr = OptimizationTrace()
  tracing = store_trace || show_trace || extended_trace
  @gdtrace
  for outiter=1:maxoutiter

    # minimize over active set
    iter = zero(T)
    lambda = one(T)
    while true
      iter += 1.
      lastVal::T = curVal
      while true
        _y_minus_ax!(tmp, y, lambda, grad_y, activeset)
        prox!(g, z, tmp, lambda, activeset)
        fz::T = value_and_gradient!(f, tmp, z, activeset)
        if fz <= _taylor_value(fy, z, y, grad_y, lambda, activeset)
          break
        end
        lambda = beta*lambda
        if lambda < beta_min
          break
        end
      end
      ωk = iter / (iter + 3.)
      _update_y!(y, z, x, ωk, activeset)
      z, x = x, z
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
    if ~add_violator!(activeset, x, g, f, tmp)
      break
    end
    fy = value_and_gradient!(f, grad_y, y, activeset)
  end
  tr
end

###





function _update_y!{T<:AbstractFloat}(
    y::StridedArray{T}, z::StridedArray{T}, x::StridedArray{T}, ω::T
    )
  @assert size(y) == size(z) == size(x)
  @inbounds for ind in eachindex(y)
    y[ind] = (1.+ω)*z[ind] - ω*x[ind]
  end
  y
end
