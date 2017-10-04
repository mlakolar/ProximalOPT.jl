###################################
# tracing based on optim package
# removed grad norm field
#

function update!(tr::OptimizationTrace{T},
                 iteration::Int64,
                 f_x::Real,
                 g_x::Real,
                 dxx::Real,   # |dx|/|x|
                 L::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool,
                 printEvery::Int64,
                 callback) where T

    os = OptimizationState{T}(iteration, f_x + g_x, dxx, L, dt)
    if store_trace
        push!(tr, os)
    end
    if show_trace && mod(iteration, printEvery) == 0
        show(os)
    end

    if callback != nothing && (iteration % printEvery == 0)
        if store_trace
            stopped = callback(tr)
        else
            stopped = callback(os)
        end
    else
        stopped = false
    end
    stopped
end


function trace!(tr, f, g, state, iteration, method::Union{ProximalGradientDescent}, options)
    dt = Dict()
    if options.extended_trace && iteration > 0
        dt["x"] = copy(state.x)
        dt["grad_f(x)"] = copy(state.grad_f_x)
    end
    update!(tr,
            iteration,
            state.f_x,
            state.g_x,
            norm_diff(state.x, state.x_previous) / vecnorm(state.x),
            state.L,
            dt,
            options.store_trace,
            options.show_trace,
            options.printEvery,
            options.callback)
end


###################################

f_residual(f_x, f_x_previous, f_tol) =
  abs(f_x - f_x_previous) / (abs(f_x) + f_tol)

function g_residual(grad_f_x, deltaX, L::T) where T
  residual = zero(T)
  @inbounds @simd for i in eachindex(deltaX)
   residual = max(residual, abs(grad_f_x[i] + deltaX[i] * L))
  end
  residual
end

function convergence_assessment(
  state::Union{ProximalGradientDescentState{T}},
  f, g, options::ProximalOptions
  ) where T

  x_converged, f_converged, grad_converged, f_increased = false, false, false, false

  norm_x = vecnorm( state.x )
  norm_dx = vecnorm( state.deltaX )

  if norm_dx < options.xtol * max( norm_x, 1. )
    x_converged = true
  end

  # Relative Tolerance
  fval = state.f_x + state.g_x
  fval_prev = state.f_x_previous + state.g_x_previous
  if f_residual(fval, fval_prev, options.ftol) < options.ftol ||
        abs(fval - fval_prev) < eps(abs(fval)+abs(fval_prev))
    f_converged = true
  end

  if fval > fval_prev
    f_increased = true
  end

  # check gradient convergence
  # this is based on FASTA paper
  residual = g_residual(state.grad_f_x, state.deltaX, state.L)
  state.maxResidual = max(state.maxResidual, residual)
  normalizer = max(vecnorm(state.grad_f_x, Inf), vecnorm( state.deltaX, Inf) * state.L) + options.gradtol
  nResidual = residual / normalizer
  grad_converged = residual / (state.maxResidual + options.gradtol) < options.gradtol || nResidual < options.gradtol

  converged = x_converged || f_converged || grad_converged

  return x_converged, f_converged, grad_converged, converged, f_increased
end

##############################
