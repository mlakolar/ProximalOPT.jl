
function solve!(
    initial_x::AbstractArray,
    f::D,
    g::P,
    method::M,
    options::ProximalOptions = ProximalOptions()
    ) where {D<:DifferentiableFunction, P<:ProximableFunction, M<:ProximalSolver}

  t0 = time()
  state = initial_state(method, options, f, g, initial_x)

  tr = OptimizationTrace{typeof(method)}()
  tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
  stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
  f_limit_reached, grad_f_limit_reached, g_limit_reached = false, false, false
  x_converged, fg_converged, grad_converged, f_increased = false, false, false, false

  converged = false
  iteration = 0

  options.show_trace && print_header(method, options)

  while !converged && !stopped && iteration < options.maxiter
    iteration += 1

    update_state!(f, g, state, method)

    x_converged, fg_converged,
    grad_converged, converged, f_increased = convergence_assessment(state, f, g, options)

    if tracing
      # update trace; callbacks can stop routine early by returning true
      stopped_by_callback = trace!(tr, f, g, state, iteration, method, options)
    end

    stopped_by_time_limit = time()-t0 > options.time_limit ? true : false
    # f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
    # grad_f_limit_reached = options.f_calls_limit > 0 && f_calls(d) >= options.f_calls_limit ? true : false
    # g_limit_reached = options.g_calls_limit > 0 && g_calls(d) >= options.g_calls_limit ? true : false

    if (f_increased && !options.allow_f_increases) || stopped_by_callback ||
        stopped_by_time_limit || f_limit_reached || grad_f_limit_reached || g_limit_reached
        stopped = true
    end
  end # while

  return OptimizationResults(
                method,
                state.x,
                state.f_x + state.g_x,
                iteration,
                iteration == options.maxiter,
                x_converged,
                options.xtol,
                norm_diff(state.x, state.x_previous, 2.),
                fg_converged,
                options.ftol,
                f_residual(state.f_x + state.g_x, state.f_x_previous + state.g_x_previous, options.ftol),
                grad_converged,
                options.gradtol,
                g_residual(state.grad_f_x, state.deltaX, state.L),
                f_increased,
                tr,
                0,
                0,
                0)

end
