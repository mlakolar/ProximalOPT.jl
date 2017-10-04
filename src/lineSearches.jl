module LineSearches

using Parameters
using ProximalBase

export
  backtrack!,has_updated_gradient,
  BackTracking


#################################
#
# Static
#
#################################

#################################
#
# TFOCS Backtrack
#
#################################

@with_kw mutable struct BackTracking{TF, TI}
    α::TF                  = 0.9
    β::TF                  = 0.5
    Lexact::TF             = Inf
    backtrack_steps::TI    = 0
    backtrack_simple::Bool = true
end

function backtrack!(state, ls::BackTracking{TF, TI}, f, g) where TF <: Real where TI <: Integer

    # Quick exit if no progress made
    @. state.deltaX = state.x - state.x_previous
    xy_sq = vecnorm( state.deltaX )^2

    if xy_sq == zero(TF)
      return true
    end

    localL = Inf
    # Compute Lipschitz estimate
    if ls.backtrack_simple
        q_x = state.f_x_previous + dot( state.deltaX, state.grad_f_x_previous ) + 0.5 * state.L * xy_sq
        localL = state.L + 2. * max( state.f_x - q_x, zero(TF) ) / xy_sq
        if abs( state.f_x_previous - state.f_x ) >= 1e-10 * max( abs( state.f_x ), abs( state.f_x_previous ) )
            ls.backtrack_simple = false
            value_and_gradient!(f, state.grad_f_x, state.x)
        end
    else
        value_and_gradient!(f, state.grad_f_x, state.x)
        localL = zero(TF)
        @inbounds @simd for i in eachindex(state.grad_f_x)
            localL += (state.grad_f_x[i] - state.grad_f_x_previous[i]) * state.deltaX[i]
        end
        localL *= 2. / xy_sq
    end

    # Exit if Lipschitz criterion satisfied, or if we hit Lexact
    ls.backtrack_steps += 1;
    (localL <= state.L || state.L >= ls.Lexact) && return true
    (ls.backtrack_steps > 20) && return true                ## TODO
    state.L = min( ls.Lexact, localL )
    state.L = min( ls.Lexact, max( localL, state.L / ls.β ) )

    return false
end

has_updated_gradient(ls::BackTracking) = ls.backtrack_simple == false

end
