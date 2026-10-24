# Run in a package environment on a compute node. This bounded comparison
# times 60 warmed iterations, with the image model and with the same value /
# gradient oracles hidden behind a passthrough wrapper. No optional packages.
using LinearAlgebra, Random, Printf, ProximalBase, ProximalSolvers
const PS = ProximalSolvers
BLAS.set_num_threads(1)

struct Uncached{F} <: AbstractFunction
    f::F
end
ProximalBase.issmooth(::Type{<:Uncached}) = true
(f::Uncached)(x) = f.f(x)
ProximalBase.value_and_gradient!(G, f::Uncached, x) = value_and_gradient!(G, f.f, x)

function block!(state, f, g, method, niter)
    fill!(state.x, 0)
    PS.reinitialize_state!(state, f, g, method)
    for _ in 1:niter
        PS.update_state!(f, g, state, method)
    end
    state
end

function measure(f, g, method, p; niter = 60)
    state = PS.initial_state(method, ProximalOptions(), f, g, zeros(p))
    block!(state, f, g, method, niter)
    bytes = @allocated block!(state, f, g, method, niter)
    samples = [@elapsed block!(state, f, g, method, niter) for _ in 1:7]
    sort!(samples)
    (time = samples[4] / niter, bytes = bytes / niter,
     objective = state.f_x + state.g_x, grads = state.grad_calls, prox = state.prox_calls)
end

function main()
    rng = Xoshiro(852)
    n, p = 1000, 400
    X = randn(rng, n, p)
    β = [ones(10); zeros(p - 10)]
    y = X * β + randn(rng, n)
    binary = Float64.(rand(rng, n) .< 0.5)
    counts = Float64.(rand(rng, 0:3, n))
    weights, offset = rand(rng, n) .+ 0.5, 0.1 .* randn(rng, n)
    cases = (("least_squares", (1 / n) * LeastSquaresLoss(y, X), 1e-3),
             ("least_squares_fallback", (1 / n) * LeastSquaresLoss(y, X), 1e3),
             ("logistic", (1 / n) * LogisticLoss(binary, X; weights, offset), 1e-3),
             ("poisson", (1 / n) * PoissonLoss(counts, X / 4; offset), 1e-3))
    println("case,cached_ns_per_iteration,uncached_ns_per_iteration,cached_bytes_per_iteration,uncached_bytes_per_iteration,cached_objective,uncached_objective,cached_grad_calls,uncached_grad_calls,cached_prox_calls,uncached_prox_calls")
    for (name, f, δ) in cases
        method, g = MonotoneAccProxGradDescent(; δ), NormL1(0.01)
        cached = measure(f, g, method, p)
        uncached = measure(Uncached(f), g, method, p)
        @assert isapprox(cached.objective, uncached.objective; rtol = 1e-7, atol = 1e-10)
        @printf "%s,%.0f,%.0f,%.1f,%.1f,%.12g,%.12g,%d,%d,%d,%d\n" name (1e9 * cached.time) (1e9 * uncached.time) cached.bytes uncached.bytes cached.objective uncached.objective cached.grads uncached.grads cached.prox uncached.prox
    end
end

main()
