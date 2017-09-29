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


function common_trace!(tr, d, state, iteration, method::Union{LBFGS, AcceleratedGradientDescent, GradientDescent, MomentumGradientDescent, ConjugateGradient}, options)
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



function check_optim_done{T<:AbstractFloat}(iter,
                                            curval::T, lastval::T,
                                            x::StridedArray{T}, z::StridedArray{T},
                                            options::ProximalOptions)
  iter >= options.maxiter || abs(curval-lastval) < convert(T, options.ftol) || _l2diff(z, x) < convert(T, options.xtol)
end



##############################

macro def(name, definition)
  esc(quote
    macro $name()
      esc($(Expr(:quote, definition)))
    end
  end)
end

@def add_linesearch_fields begin
    x_ls::Array{T,N}
    alpha::T
    mayterminate::Bool
    lsr::LineSearches.LineSearchResults
end

@def initial_linesearch begin
    (similar(initial_x), # Buffer of x for line search in state.x_ls
    LineSearches.alphainit(one(T), initial_x, gradient(d), value(d)), # Keep track of step size in state.alpha
    false, # state.mayterminate
    LineSearches.LineSearchResults(T))
end
