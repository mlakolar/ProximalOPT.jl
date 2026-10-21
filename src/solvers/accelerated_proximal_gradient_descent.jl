

# implements the accelerated proximal gradient method in section 4.3 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
#
# The momentum is restarted whenever the objective increases
# (O'Donoghue and Candès, 2015, Adaptive restart for accelerated gradient schemes),
# so the objective is nonincreasing across iterations.

"""
    AccProxGradDescent(; β = 0.5)

Accelerated proximal gradient descent (Parikh and Boyd, *Proximal Algorithms*,
§4.3). Each iteration extrapolates `y = x + k/(k+3) (x - x')` and takes a
proximal gradient step from `y`; the step size `1/L` is found by
backtracking, shrinking it by the factor `β ∈ (0, 1)`.

The momentum restarts whenever the objective would increase (O'Donoghue and
Candès, *Adaptive restart for accelerated gradient schemes*, 2015), so the
objective is nonincreasing. Use with [`solve!`](@ref).
"""
struct AccProxGradDescent{T<:AbstractFloat} <: ProximalSolver
  β::T          # factor by which the step size 1/L shrinks during backtracking
end

Base.summary(::AccProxGradDescent) = "Accelerated Proximal Gradient Descent"

function AccProxGradDescent(; β::AbstractFloat = 0.5)
  0 < β < 1 || error("β must be in (0, 1).")
  AccProxGradDescent(β)
end

mutable struct AccProxGradDescentState{T,N}
    x::Array{T,N}
    f_x::T
    g_x::T
    x_previous::Array{T,N}
    f_x_previous::T
    g_x_previous::T
    y::Array{T,N}
    f_y::T
    grad_f_y::Array{T,N}
    xhat::Array{T,N}
    deltaX::Array{T,N}
    L::T
    maxResidual::T
    k::Int            # iterations since the last restart
    restarts::Int
    backtrack_steps::Int
end


function initial_state(
          method::AccProxGradDescent,
          options,
          f,
          g,
          x0::Array{T}) where T

    AccProxGradDescentState(
                         x0,          # Maintain current state in state.x
                         value(f, x0),
                         value(g, x0),
                         copy(x0),    # x_previous = x0, so the first step has no momentum
                         T(NaN),
                         T(NaN),
                         similar(x0), # extrapolated point y
                         T(NaN),      # f(y)
                         similar(x0), # gradient of f at y
                         similar(x0), # Temporary storage
                         similar(x0), # Temporary storage for x1 - x0
                         one(T),      # L
                         T(-Inf),     # maxResidual
                         0,
                         0,
                         0            # backtrack_steps
                         )
end

function update_state!(
    f, g,
    state::AccProxGradDescentState{T},
    method::AccProxGradDescent
    ) where T

    ω = T(state.k / (state.k + 3))
    @. state.y = state.x + ω * (state.x - state.x_previous)

    # copy x --> x_previous
    copyto!(state.x_previous, state.x)
    state.f_x_previous = state.f_x
    state.g_x_previous = state.g_x

    while true
      _prox_grad_step!(f, g, state, method)

      # a step from y = x_previous cannot increase the objective,
      # so this restarts at most once
      if ω > 0 && state.f_x + state.g_x > state.f_x_previous + state.g_x_previous
        ω = zero(T)
        state.k = 0
        state.restarts += 1
        copyto!(state.y, state.x_previous)
      else
        break
      end
    end

    state.k += 1
    @. state.deltaX = state.x - state.x_previous
    state
end

# x = prox_{g/L}(y - ∇f(y)/L), increasing L until
# f(x) <= f(y) + <∇f(y), x - y> + L/2 ‖x - y‖²
function _prox_grad_step!(f, g, state::AccProxGradDescentState{T}, method::AccProxGradDescent) where T
    state.f_y = value_and_gradient!(f, state.grad_f_y, state.y)
    while true
      @. state.xhat = state.y - state.grad_f_y / state.L
      prox!(g, state.x, state.xhat, one(T) / state.L)
      state.f_x = value(f, state.x)

      @. state.xhat = state.x - state.y
      upper = state.f_y + dot(state.grad_f_y, state.xhat) + state.L / 2 * sum(abs2, state.xhat)
      state.backtrack_steps += 1
      # slack guards against an endless loop from rounding error near the solution
      state.f_x <= upper + 10 * eps(T) * max(abs(state.f_y), one(T)) && break
      state.L /= method.β
    end
    state.g_x = value(g, state.x)
end
