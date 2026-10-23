

# implements the accelerated proximal gradient method in section 4.3 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
#
# The momentum is restarted whenever the objective increases
# (O'Donoghue and Candès, 2015, Adaptive restart for accelerated gradient schemes),
# so the objective is nonincreasing across iterations.

"""
    AccProxGradDescent(; β = 0.5, α = 0.9)

Accelerated proximal gradient descent (Parikh and Boyd, *Proximal Algorithms*,
§4.3). Each iteration extrapolates `y = x + k/(k+3) (x - x')` and takes a
proximal gradient step from `y`; the step size `1/L` is found by
backtracking, shrinking it by the factor `β ∈ (0, 1)`. Before each iteration
`L` is multiplied by `α ∈ (0, 1]`, so that the step grows again where the
curvature is smaller, as in TFOCS; `α = 1` keeps `L` from decreasing. On the
lasso of the benchmark, `α = 0.9` takes 1.2 to 6.5 times less time than
`α = 1`.

The momentum restarts whenever the objective would increase (O'Donoghue and
Candès, *Adaptive restart for accelerated gradient schemes*, 2015), so the
objective is nonincreasing. Use with [`solve!`](@ref).
"""
struct AccProxGradDescent{T<:AbstractFloat} <: ProximalSolver
  β::T          # factor by which the step size 1/L shrinks during backtracking
  α::T          # factor by which L shrinks before each iteration, so the step can grow
end

Base.summary(::AccProxGradDescent) = "Accelerated Proximal Gradient Descent"

function AccProxGradDescent(; β::AbstractFloat = 0.5, α::AbstractFloat = 0.9)
  0 < β < 1 || error("β must be in (0, 1).")
  0 < α <= 1 || error("α must be in (0, 1].")
  AccProxGradDescent(promote(β, α)...)
end

# the iterates have the type of the starting point x0; the gradient is dense
mutable struct AccProxGradDescentState{T, X<:AbstractArray{T}, G<:AbstractArray{T}}
    x::X
    f_x::T
    g_x::T
    x_previous::X
    f_x_previous::T
    g_x_previous::T
    y::X
    f_y::T
    grad_f_y::G
    xhat::X
    deltaX::X
    L::T
    maxResidual::T
    k::Int            # iterations since the last restart
    restarts::Int
    backtrack_steps::Int
    f_calls::Int
    grad_calls::Int
    prox_calls::Int
end


function initial_state(
          method::AccProxGradDescent,
          options,
          f,
          g,
          x0::AbstractArray{T}) where T

    AccProxGradDescentState(
                         x0,          # Maintain current state in state.x
                         f(x0),
                         g(x0),
                         copy(x0),    # x_previous = x0, so the first step has no momentum
                         T(NaN),
                         T(NaN),
                         similar(x0), # extrapolated point y
                         T(NaN),      # f(y)
                         Array{T}(undef, size(x0)), # gradient of f at y
                         similar(x0), # Temporary storage
                         similar(x0), # Temporary storage for x1 - x0
                         one(T),      # L
                         T(-Inf),     # maxResidual
                         0,
                         0,
                         0,           # backtrack_steps
                         1,           # f_calls
                         0,           # grad_calls
                         0            # prox_calls
                         )
end

function update_state!(
    f, g,
    state::AccProxGradDescentState{T},
    method::AccProxGradDescent
    ) where T

    ω = T(state.k / (state.k + 3))
    state.L *= T(method.α)
    @. state.y = state.x + ω * (state.x - state.x_previous)
    _drop!(state.y)

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
    state.f_y = value_and_gradient!(state.grad_f_y, f, state.y)
    state.f_calls += 1
    state.grad_calls += 1
    while true
      @. state.xhat = state.y - state.grad_f_y / state.L
      state.g_x = value_and_prox!(state.x, g, state.xhat, one(T) / state.L)
      _drop!(state.x)
      state.f_x = f(state.x)
      state.prox_calls += 1
      state.f_calls += 1

      @. state.xhat = state.x - state.y
      upper = state.f_y + dot(state.grad_f_y, state.xhat) + state.L / 2 * sum(abs2, state.xhat)
      state.backtrack_steps += 1
      # slack guards against an endless loop from rounding error near the solution
      state.f_x <= upper + 10 * eps(T) * max(abs(state.f_y), one(T)) && break
      state.L /= method.β
    end
    state
end
