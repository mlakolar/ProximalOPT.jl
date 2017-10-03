
function solve!(
    initial_x::AbstractArray,
    f::D,
    g::P,
    method::M,
    options::ProximalOptions = ProximalOptions(),
    state = initial_state(method, options, d, complex_to_real(d, initial_x))
    ) where {D<:DifferentiableFunction, P<:ProximableFunction, M<:ProximalSolver}

    t0 = time()

    n = length(initial_x)
    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, grad_f_limit_reached, g_limit_reached = false, false, false
    x_converged, fg_converged, f_increased = false, false, false

    converged = false
    iteration = 0

    options.show_trace && print_header(method, options)
    trace!(tr, f, g, state, iteration, method, options)

    while !converged && !stopped && iteration < options.maxiter
        iteration += 1

        update_state!(f, g, state, method)

        x_converged, fg_converged,
        g_converged, converged, f_increased = assess_convergence(state, f, g, options)

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

    return MultivariateOptimizationResults(method,
                                            NLSolversBase.iscomplex(d),
                                            initial_x,
                                            f_increased ? state.x_previous : state.x,
                                            f_increased ? state.f_x_previous : value(d),
                                            iteration,
                                            iteration == options.iterations,
                                            x_converged,
                                            options.x_tol,
                                            x_residual(state.x, state.x_previous),
                                            fg_converged,
                                            options.f_tol,
                                            f_residual(value(d), state.f_x_previous, options.f_tol),
                                            g_converged,
                                            options.g_tol,
                                            g_residual(gradient(d)),
                                            f_increased,
                                            tr,
                                            f_calls(d),
                                            g_calls(d),
                                            h_calls(d))

    end

end
