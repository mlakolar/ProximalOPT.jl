

# minimizes f(x) + g(x), where both f and g are proximable, using the
# over-relaxed scaled ADMM of sections 3.1.1, 3.3 and 3.4.3 of
# Boyd et al. (2011), Distributed Optimization and Statistical Learning
# via the Alternating Direction Method of Multipliers

"""
    ADMMSolver(; ρ = 1.0, α = 1.0, abstol = 1e-6, reltol = 1e-4)

The alternating direction method of multipliers for `f(x) + g(x)` when both
`f` and `g` have easy proximal operators (Boyd et al., *Distributed
Optimization and Statistical Learning via the Alternating Direction Method of
Multipliers*, 2011). It splits `x = z` and iterates

    x ← prox_{f/ρ}(z - u)
    z ← prox_{g/ρ}(α x + (1 - α) z + u)
    u ← u + α x + (1 - α) z_previous - z

`ρ > 0` is the augmented Lagrangian parameter and `α ∈ (0, 2)` the
over-relaxation (`α = 1` is plain ADMM). It stops when the primal residual
`‖x - z‖` and dual residual `ρ‖z - z_previous‖` are below
`√n abstol + reltol * scale` (§3.3 of the paper). Of the
[`ProximalOptions`](@ref), only `maxiter`, `time_limit`, `show_trace` and
`printEvery` are used. Use with [`solve!`](@ref).
"""
struct ADMMSolver{T<:AbstractFloat} <: ProximalSolver
  ρ::T          # augmented Lagrangian parameter
  α::T          # over-relaxation parameter, α ∈ (0, 2); α = 1 is plain ADMM
  abstol::T
  reltol::T
end

Base.summary(::ADMMSolver) = "ADMM"

function ADMMSolver(; ρ::AbstractFloat = 1.0, α::AbstractFloat = 1.0,
                    abstol::AbstractFloat = 1e-6, reltol::AbstractFloat = 1e-4)
  ρ > 0 || error("ρ must be positive.")
  0 < α < 2 || error("α must be in (0, 2).")
  ADMMSolver(promote(ρ, α, abstol, reltol)...)
end

"""
    ADMMResults

Returned by [`solve!`](@ref) with an [`ADMMSolver`](@ref): `minimizer`,
`minimum` (`f + g` at the minimizer), `iterations`, `converged`, and the final
primal and dual residuals `r_norm`, `s_norm` with their tolerances `eps_pri`,
`eps_dual`.
"""
struct ADMMResults{T,N}
    minimizer::Array{T,N}
    minimum::T
    iterations::Int
    converged::Bool
    r_norm::T     # primal residual ‖x - z‖
    s_norm::T     # dual residual ρ‖z - z_previous‖
    eps_pri::T
    eps_dual::T
end

Base.summary(::ADMMResults) = "ADMM"
minimizer(r::ADMMResults) = r.minimizer
minimum(r::ADMMResults) = r.minimum
iterations(r::ADMMResults) = r.iterations
converged(r::ADMMResults) = r.converged

function Base.show(io::IO, r::ADMMResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" summary(r)
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * primal residual %.2e (tolerance %.2e)\n" r.r_norm r.eps_pri
    @printf io "   * dual residual   %.2e (tolerance %.2e)\n" r.s_norm r.eps_dual
    return
end

# x is the starting point for z and is overwritten with the solution
function solve!(
    x::Array{T},
    f::ProximableFunction,
    g::ProximableFunction,
    method::ADMMSolver,
    options::ProximalOptions = ProximalOptions()
    ) where {T<:AbstractFloat}

  t0 = time()
  ρ, α = T(method.ρ), T(method.α)
  γ = one(T) / ρ
  sqrt_n = sqrt(T(length(x)))

  z = x
  xk = copy(x)
  u = zero(x)
  z_previous = similar(x)
  tmp = similar(x)

  r_norm = s_norm = eps_pri = eps_dual = T(Inf)
  converged = false
  iteration = 0

  options.show_trace && @printf "Iter     Function value   primal res.      dual res.\n"

  while !converged && iteration < options.maxiter && time() - t0 <= options.time_limit
    iteration += 1

    # x-update: x = prox_{f/ρ}(z - u)
    @. tmp = z - u
    prox!(f, xk, tmp, γ)

    # z-update with relaxation: z = prox_{g/ρ}(α x + (1 - α) z + u)
    copyto!(z_previous, z)
    @. tmp = α * xk + (one(T) - α) * z_previous + u
    prox!(g, z, tmp, γ)

    # u-update
    @. u = tmp - z

    # check convergence
    r_norm = norm_diff(xk, z)
    s_norm = ρ * norm_diff(z, z_previous)
    eps_pri = sqrt_n * T(method.abstol) + T(method.reltol) * max(norm(xk), norm(z))
    eps_dual = sqrt_n * T(method.abstol) + T(method.reltol) * ρ * norm(u)
    converged = r_norm <= eps_pri && s_norm <= eps_dual

    if options.show_trace && iteration % options.printEvery == 0
      @printf "%6d   %14e   %14e   %14e\n" iteration value(f, z) + value(g, z) r_norm s_norm
    end
  end

  ADMMResults(z, value(f, z) + value(g, z), iteration, converged,
              r_norm, s_norm, eps_pri, eps_dual)
end
