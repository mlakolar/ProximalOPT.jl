

# implements the algorithm in section 4.2 of
# https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

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
                         -Inf
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

      # check if enough progress is done
      backtrack!(state, method.ls, f, g) && break
    end
    if has_updated_gradient( method.ls ) == false
        value_and_gradient!(f, state.grad_f_x, state.x)
    end
end
