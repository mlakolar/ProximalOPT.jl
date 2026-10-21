

# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

"""
    ProximalGradientDescent(; ls = LineSearches.BackTracking())

Proximal gradient descent (Parikh and Boyd, *Proximal Algorithms*, §4.2):
`x ← prox_{g/L}(x - ∇f(x) / L)`, with the step size `1/L` chosen by the
backtracking line search `ls` of TFOCS. Use with [`solve!`](@ref).
"""
struct ProximalGradientDescent{L} <: ProximalSolver
  ls::L
end

Base.summary(::ProximalGradientDescent) = "Proximal Gradient Descent"

ProximalGradientDescent(; ls = LineSearches.BackTracking()) =
  ProximalGradientDescent{typeof(ls)}(ls)

mutable struct ProximalGradientDescentState{T,N}
    x::Array{T,N}
    f_x::T
    g_x::T
    grad_f_x::Array{T,N}
    x_previous::Array{T,N}
    f_x_previous::T
    g_x_previous::T
    grad_f_x_previous::Array{T,N}
    xhat::Array{T,N}
    deltaX::Array{T,N}
    L::T
    maxResidual::T
    backtrack_simple::Bool  # line search uses function values, not gradients
    backtrack_steps::Int
    f_calls::Int
    grad_calls::Int
    prox_calls::Int
end


function initial_state(
          method::ProximalGradientDescent,
          options,
          f,
          g,
          x0::Array{T}) where T

    grad_f_x = similar(x0)
    f_x = value_and_gradient!(f, grad_f_x, x0)
    g_x = value(g, x0)

    ProximalGradientDescentState(
                         x0,          # Maintain current state in state.x
                         f_x,         # Maintain current f(x) in state.f_x
                         g_x,         # Maintain current g(x) in state.g_x
                         grad_f_x,    # Maintain gradient of f at current x in state.grad_f_x
                         similar(x0), # Maintain previous state in state.x_previous
                         T(NaN),      # Store previous f in state.f_x_previous
                         T(NaN),      # Store previous g in state.g_x_previous
                         similar(x0), # Maintain previous gradient of f in state.grad_f_x_previous for convergence check
                         similar(x0), # Temporary storage
                         similar(x0), # Temporary storage for x1 - x0
                         one(T),      # L
                         -Inf,
                         true,        # backtrack_simple
                         0,           # backtrack_steps
                         1,           # f_calls
                         1,           # grad_calls
                         0            # prox_calls
                         )
end

function update_state!(
    f, g,
    state::ProximalGradientDescentState{T},
    method::ProximalGradientDescent
    ) where T

    # copy x --> x_previous
    copy!(state.x_previous, state.x)
    state.f_x_previous = state.f_x
    state.g_x_previous = state.g_x
    copy!(state.grad_f_x_previous, state.grad_f_x)
    state.L *= method.ls.α

    # backtracking loop
    while true
      # find next point
      @. state.xhat =  state.x_previous - state.grad_f_x_previous / state.L
      prox!(g, state.x, state.xhat, one(T) / state.L)
      state.f_x = value(f, state.x)
      state.g_x = value(g, state.x)
      state.prox_calls += 1
      state.f_calls += 1

      # check if enough progress is done
      backtrack!(state, method.ls, f, g) && break
    end
    if has_updated_gradient( state, method.ls ) == false
        value_and_gradient!(f, state.grad_f_x, state.x)
        state.f_calls += 1
        state.grad_calls += 1
    end
end
