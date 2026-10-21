module LineSearches

using LinearAlgebra
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

"""
    BackTracking(; α = 0.9, β = 0.5, Lexact = Inf)

The backtracking line search of TFOCS (`private/tfocs_backtrack.m`). Each
iteration starts from `L ← α L`, where `1/L` is the step size, and while the
local Lipschitz estimate `L̂` of the step exceeds `L` it retries with
`L ← min(Lexact, max(L̂, L / β))`. `L̂` comes from function values until they
are too close to compare in floating point, and from gradients after that.

The object holds settings only and can be shared between solves; the
bookkeeping lives in the solver state.
"""
Base.@kwdef struct BackTracking{T<:Real}
    α::T      = 0.9
    β::T      = 0.5
    Lexact::T = Inf
end

function backtrack!(state, ls::BackTracking{T}, f, g) where T

    # Quick exit if no progress made
    @. state.deltaX = state.x - state.x_previous
    xy_sq = norm( state.deltaX )^2

    if xy_sq == zero(T)
      return true
    end

    localL = Inf
    # Compute Lipschitz estimate
    if state.backtrack_simple
        q_x = state.f_x_previous + dot( state.deltaX, state.grad_f_x_previous ) + 0.5 * state.L * xy_sq
        localL = state.L + 2. * max( state.f_x - q_x, zero(T) ) / xy_sq
        # as in TFOCS, keep the function value estimate while f(x) and f(x') are far
        # enough apart to compare in floating point, then switch for good to the
        # gradient estimate, which needs ∇f(x)
        state.backtrack_simple = abs( state.f_x_previous - state.f_x ) >= 1e-10 * max( abs( state.f_x ), abs( state.f_x_previous ) )
        if !state.backtrack_simple
            value_and_gradient!(f, state.grad_f_x, state.x)
        end
    else
        value_and_gradient!(f, state.grad_f_x, state.x)
        localL = zero(T)
        @inbounds @simd for i in eachindex(state.grad_f_x)
            localL += (state.grad_f_x[i] - state.grad_f_x_previous[i]) * state.deltaX[i]
        end
        localL *= 2. / xy_sq
    end

    # Exit if Lipschitz criterion satisfied, or if we hit Lexact
    state.backtrack_steps += 1;
    (localL <= state.L || state.L >= ls.Lexact) && return true
    state.L = min( ls.Lexact, max( localL, state.L / ls.β ) )

    return false
end

has_updated_gradient(state, ::BackTracking) = state.backtrack_simple == false

end
