# ProximalBase functions as the terms of ProximalAlgorithms.jl (0.7, built on
# ProximalCore 0.1), so that both packages' solvers run on the same objects.
# Nothing here is a dependency of ProximalSolvers.

using ProximalAlgorithms, ProximalCore, ProximalBase

# The smooth term. ProximalAlgorithms calls value_and_gradient(f, x), which
# returns a new gradient array at every call; that allocation is part of its
# interface and is counted in the measurements.
struct PASmooth{F}
  f::F
end
(w::PASmooth)(x) = w.f(x)
function ProximalAlgorithms.value_and_gradient(w::PASmooth, x)
  g = similar(x)
  v = ProximalBase.value_and_gradient!(g, w.f, x)
  return v, g
end

# The proximable term. ProximalCore's prox! returns the value at the result,
# which is what ProximalBase's value_and_prox! computes.
struct PAProx{G}
  g::G
end
(w::PAProx)(x) = w.g(x)
ProximalCore.prox!(y, w::PAProx, x, γ) = ProximalBase.value_and_prox!(y, w.g, x, γ)
ProximalCore.is_convex(::Type{PAProx{G}}) where {G} = ProximalBase.isconvex(G)

# Runs an algorithm of ProximalAlgorithms with its default stopping test at
# tolerance `tol`, and returns the solution and the number of iterations.
# `maxit` and `time_limit` cap the run, as maxiter and time_limit do for
# ProximalSolvers: a run whose iterates turn NaN never passes the test.
const TIME_LIMIT = Ref(60.0)       # seconds per run, for every solver

function run_pa(alg, tol; maxit = 1_000_000, time_limit = TIME_LIMIT[], kwargs...)
  t0 = time()
  stop(iter, state) = ProximalAlgorithms.default_stopping_criterion(tol, iter, state) || time() - t0 > time_limit
  alg(; maxit, stop)(; kwargs...)
end

# The same for exactly `k` iterations (no stopping test), to measure the bytes
# allocated per iteration.
run_pa_iters(alg, k; kwargs...) = alg(; maxit = k, stop = (iter, state) -> false)(; kwargs...)
