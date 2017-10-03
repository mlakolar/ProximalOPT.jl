###################################
# tracing based on optim package
# removed grad norm field
#

function update!(tr::OptimizationTrace{T},
                 iteration::Int64,
                 f_x::Real,
                 dt::Dict,
                 store_trace::Bool,
                 show_trace::Bool,
                 printEvery::Int64,
                 callback) where T

    os = OptimizationState{T}(iteration, f_x, dt)
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
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end


###################################

f_residual(f_x, f_x_previous, f_tol) = abs(f_x - f_x_previous) / (abs(f_x) + f_tol)

function convergence_assessment(
    state::Union{ProximalGradientDescentState}, f, g, options)

    x_converged, f_converged, f_increased = false, false, false

    norm_x = vecnorm( state.x )
    norm_dx = norm_diff(state.x, state.x_previous, 2.)


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

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end

##############################
