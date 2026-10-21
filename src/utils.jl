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

    if callback !== nothing && (iteration % printEvery == 0)
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


function trace!(tr, f, g, state, iteration, method::ProximalSolver, options)
    dt = Dict()
    if options.extended_trace && iteration > 0
        dt["x"] = copy(state.x)
        extended_trace!(dt, state)
    end
    update!(tr,
            iteration,
            state.f_x,
            state.g_x,
            norm_diff(state.x, state.x_previous) / norm(state.x),
            state.L,
            dt,
            options.store_trace,
            options.show_trace,
            options.printEvery,
            options.callback)
end

extended_trace!(dt, state::ProximalGradientDescentState) = dt["grad_f(x)"] = copy(state.grad_f_x)
extended_trace!(dt, state::AccProxGradDescentState) = dt["grad_f(y)"] = copy(state.grad_f_y)


###################################

f_residual(f_x, f_x_previous, f_tol) =
  abs(f_x - f_x_previous) / (abs(f_x) + f_tol)

# FASTA residual r = ∇f(x) + (xhat - x) / τ, where xhat = x_previous - τ∇f(x_previous)
# and τ = 1/L, that is r = ∇f(x) - ∇f(x_previous) - L (x - x_previous).
# Since L (xhat - x) ∈ ∂g(x), r ∈ ∇f(x) + ∂g(x) and r = 0 at a minimizer.
function g_residual(grad_f_x, grad_f_x_previous, deltaX, L::T) where T
  residual = zero(T)
  @inbounds @simd for i in eachindex(deltaX)
   residual = max(residual, abs(grad_f_x[i] - grad_f_x_previous[i] - deltaX[i] * L))
  end
  residual
end

g_residual(state::ProximalGradientDescentState) =
  g_residual(state.grad_f_x, state.grad_f_x_previous, state.deltaX, state.L)
# The accelerated method takes its step from y, so L (y - x) - ∇f(y) ∈ ∂g(x) and
# r = ∇f(x) - ∇f(y) - L (x - y) ∈ ∇f(x) + ∂g(x). It does not evaluate ∇f(x), but
# ‖r‖ ≤ (L + L_f) ‖x - y‖ when ∇f is L_f-Lipschitz, so L ‖x - y‖ is used instead.
function g_residual(state::AccProxGradDescentState{T}) where T
  residual = zero(T)
  @inbounds @simd for i in eachindex(state.x)
   residual = max(residual, abs(state.x[i] - state.y[i]))
  end
  state.L * residual
end

function convergence_assessment(
  state::Union{ProximalGradientDescentState, AccProxGradDescentState},
  f, g, options::ProximalOptions
  )

  x_converged, f_converged, grad_converged, f_increased = false, false, false, false

  norm_x = norm( state.x )
  norm_dx = norm( state.deltaX )

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

  grad_converged = gradient_converged!(state, options)

  converged = x_converged || f_converged || grad_converged

  return x_converged, f_converged, grad_converged, converged, f_increased
end

# check gradient convergence
# this is based on FASTA paper: relative residual r / max_k r_k and
# normalized residual r / (max(‖∇f(x)‖, ‖(xhat - x) / τ‖) + gradtol)
function gradient_converged!(state::ProximalGradientDescentState{T}, options::ProximalOptions) where T
  residual = g_residual(state)
  state.maxResidual = max(state.maxResidual, residual)
  # (xhat - x) / τ = -(L (x - x_previous) + ∇f(x_previous))
  subgrad = zero(T)
  @inbounds @simd for i in eachindex(state.deltaX)
    subgrad = max(subgrad, abs(state.L * state.deltaX[i] + state.grad_f_x_previous[i]))
  end
  normalizer = max(norm(state.grad_f_x, Inf), subgrad) + options.gradtol
  nResidual = residual / normalizer
  residual / (state.maxResidual + options.gradtol) < options.gradtol || nResidual < options.gradtol
end

function gradient_converged!(state::AccProxGradDescentState{T}, options::ProximalOptions) where T
  residual = g_residual(state)
  state.maxResidual = max(state.maxResidual, residual)
  # (xhat - x) / τ = L (y - x) - ∇f(y), with xhat = y - τ∇f(y)
  subgrad = zero(T)
  @inbounds @simd for i in eachindex(state.x)
    subgrad = max(subgrad, abs(state.L * (state.y[i] - state.x[i]) - state.grad_f_y[i]))
  end
  normalizer = max(norm(state.grad_f_y, Inf), subgrad) + options.gradtol
  nResidual = residual / normalizer
  residual / (state.maxResidual + options.gradtol) < options.gradtol || nResidual < options.gradtol
end

##############################
