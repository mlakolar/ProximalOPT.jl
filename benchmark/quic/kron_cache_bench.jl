# Compare the two existing Kronecker caches on the same symmetric quadratics.
# All data generation, compilation, solves and validation belong in a Slurm
# allocation on Discovery. No extra packages beyond benchmark/quic are used.
#
# julia --project=benchmark/quic benchmark/quic/kron_cache_bench.jl \
#     --sizes 30,100,300 --rounds 4 --out /absolute/path/kron_cache.csv

using LinearAlgebra, Random, Printf
using ProximalBase, ProximalSolvers

const PS = ProximalSolvers

function option(name, default)
    i = findfirst(==(name), ARGS)
    i === nothing && return default
    i < length(ARGS) || error("$name needs an argument")
    ARGS[i + 1]
end

const SIZES = parse.(Int, split(option("--sizes", "30,100,300"), ','))
const ROUNDS = parse(Int, option("--rounds", "4"))
const OUT = abspath(option("--out", joinpath(@__DIR__, "kron_cache.csv")))
const TOL = parse(Float64, option("--tol", "1e-8"))

# The covariance design of benchmark/engine: identity versus an AR(1)
# population covariance, so the population precision difference is banded.
function sample_problem(p; n = max(600, 2p), seed = 2200 + p)
    rng = Xoshiro(seed)
    covariance_y = [0.7^abs(i - j) for i in 1:p, j in 1:p]
    X = randn(rng, n, p)
    Y = randn(rng, n, p) * cholesky(Symmetric(covariance_y)).U
    X .-= sum(X; dims = 1) ./ n
    Y .-= sum(Y; dims = 1) ./ n
    A, B = Matrix(Symmetric(X' * X / (n - 1))), Matrix(Symmetric(Y' * Y / (n - 1)))
    (; A, B, X, Y)
end

# Fixed weights of benchmark/engine's reduced symmetric inverse-Hessian
# subproblem. These are the lower-triangle entries of the gradient standard
# errors at the asymmetric starting point E_ab, transformed to svec weights.
# This deliberately measures one fixed-weight subproblem, not DiffPrecTest's
# full statistical procedure or its later reweighting.
function inverse_weights(A, B, X, Y; a = 2, b = 1)
    n, p = size(X)
    W = zeros(p, p)
    for c in 1:p, r in c:p
        constant = r == a && c == b ? -1.0 : 0.0
        target = A[r, a] * B[b, c] + constant
        q2, v2 = 0.0, 0.0
        for k in 1:n
            q = X[k, r] * X[k, a] * B[b, c] + constant
            v = A[r, a] * Y[k, b] * Y[k, c] + constant
            q2 += q * q
            v2 += v * v
        end
        variance = (q2 + v2 - 2n * target^2) / (n * (n - 1))
        # Only suppress negative roundoff in a variance, never NaN or Inf.
        isfinite(variance) || error("nonfinite inverse-Hessian weight")
        scale = (q2 + v2 + 2n * target^2) / (n * (n - 1))
        variance >= -128eps(Float64) * max(scale, floatmin(Float64)) ||
            error("negative inverse-Hessian variance beyond roundoff")
        weight = sqrt(max(variance, 0.0))
        W[r, c] = W[c, r] = r == c ? weight : weight / sqrt(2.0)
    end
    W[a, b] = W[b, a] = 0.0       # the inverse-Hessian row's free target
    W
end

function cases(p)
    d = sample_problem(p)
    A, B = d.A, d.B
    C = A - B
    lm = maximum(abs, C)
    result = Any[]
    for fraction in (0.5, 0.2)
        push!(result, (name = "dtrace", p = p, level = fraction,
                       f = KronQuadratic(A, B, C), g = NormL1(fraction * lm),
                       start = SymmetricSparseIterate(p)))
    end
    W = inverse_weights(A, B, d.X, d.Y)
    C = zeros(p, p)
    C[2, 1] = C[1, 2] = -1 / sqrt(2.0)
    start = SymmetricSparseIterate(p)
    start[2, 1] = 1 / sqrt(2.0)
    for level in (1.0, 3.0)
        push!(result, (name = "inverse_hessian_svec", p = p, level = level,
                       f = KronQuadratic(A, B, C), g = NormL1(level, W), start = start))
    end
    result
end

# Independent objective and entrywise KKT calculation, outside the timed solve.
# On the symmetric domain the smooth gradient is sym(A Θ B) + C; an
# off-diagonal coordinate changes two entries, which does not change its zero
# subgradient condition when the symmetric penalty counts both entries too.
function diagnostics(f, g, x)
    theta = Matrix(x)
    product = f.A * theta * f.B
    objective = sum(theta .* product) / 2 + sum(f.C .* theta) + g(theta)
    gradient = (product + product') / 2 + f.C
    stationarity = 0.0
    for j in eachindex(theta)
        weight = g.w === nothing ? 1.0 : g.w[j]
        penalty = g.λ0 * weight
        residual = iszero(theta[j]) ? max(abs(gradient[j]) - penalty, 0.0) :
                                      abs(gradient[j] + penalty * sign(theta[j]))
        stationarity = max(stationarity, residual)
    end
    (; theta, objective, stationarity)
end

function run_case(d, cache; seed = 2233)
    # Model and iterate construction are reported separately; total time is
    # their sum with solve!. Both variants reuse the same immutable A/B/C.
    setup = @elapsed begin
        x = copy(d.start)
        m = cache == "M=AThetaB" ? cdmodel(d.f, x) :
            PS.KronUModel(d.f.A, d.f.B, d.f.C, 1.0)
        method = CDSolver(; maxpasses = 5000, tol = TOL, order = :random,
                           momentum = false, rng = Xoshiro(seed))
    end
    solve_seconds = @elapsed r = solve!(x, m, d.g, method)
    diag = diagnostics(d.f, d.g, x)
    kkt = kkt_residual(m, d.g, x)
    (; seconds = setup + solve_seconds, setup, passes = r.passes,
       converged = r.converged, objective = diag.objective,
       stationarity = diag.stationarity, theta = diag.theta, kkt,
       nnz = count(!iszero, diag.theta))
end

median_time(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function main()
    all(>=(3), SIZES) || error("all sizes must be at least 3")
    ROUNDS >= 1 || error("--rounds must be positive")
    TOL > 0 || error("--tol must be positive")
    BLAS.set_num_threads(1)
    println("Julia ", VERSION, "; BLAS threads ", BLAS.get_num_threads())
    println("sizes=", SIZES, " rounds=", ROUNDS, " tol=", TOL)
    # Compile both cache types and both problem shapes before timing.
    for d in cases(6), cache in ("M=AThetaB", "U=ThetaB")
        run_case(d, cache)
    end
    mkpath(dirname(OUT))
    summary_path = replace(OUT, r"\.csv$" => "") * "_summary.csv"
    all_ok = true
    open(OUT, "w") do raw
        open(summary_path, "w") do summary
            println(raw, "problem,p,level,round,cache,seconds,setup_seconds,passes,converged,objective,kkt_relative,stationarity_inf,nonzero_entries,relative_solution_difference,relative_objective_difference")
            println(summary, "problem,p,level,M_best_seconds,U_best_seconds,U_over_M_best,M_median_seconds,U_median_seconds,U_over_M_median,max_solution_difference,max_objective_difference,max_kkt_relative,max_stationarity_inf,all_converged,validation_passed")
            for p in SIZES, d in cases(p)
                times = Dict(cache => Float64[] for cache in ("M=AThetaB", "U=ThetaB"))
                max_solution, max_objective, max_kkt, max_stationarity = 0.0, 0.0, 0.0, 0.0
                converged = true
                for round in 1:ROUNDS
                    # Alternate which implementation runs first; each receives
                    # the same starting point and a fresh identical RNG stream.
                    order = isodd(round) ? ("M=AThetaB", "U=ThetaB") : ("U=ThetaB", "M=AThetaB")
                    pair = Dict{String, Any}()
                    for cache in order
                        GC.gc()
                        pair[cache] = run_case(d, cache)
                        push!(times[cache], pair[cache].seconds)
                    end
                    a, b = pair["M=AThetaB"], pair["U=ThetaB"]
                    solution_difference = norm(a.theta - b.theta) / max(1.0, norm(a.theta))
                    objective_difference = abs(a.objective - b.objective) / max(1.0, abs(a.objective))
                    max_solution = max(max_solution, solution_difference)
                    max_objective = max(max_objective, objective_difference)
                    for cache in order
                        r = pair[cache]
                        max_kkt = max(max_kkt, r.kkt)
                        max_stationarity = max(max_stationarity, r.stationarity)
                        converged &= r.converged
                        println(raw, join((d.name, p, d.level, round, cache, r.seconds, r.setup,
                            r.passes, r.converged, r.objective, r.kkt, r.stationarity, r.nnz,
                            solution_difference, objective_difference), ','))
                    end
                    flush(raw)
                end
                mt, ut = times["M=AThetaB"], times["U=ThetaB"]
                valid = converged && max_solution <= 5e-5 && max_objective <= 1e-8 &&
                        max_kkt <= max(100TOL, 1e-6) && max_stationarity <= max(100TOL, 1e-6)
                all_ok &= valid
                println(summary, join((d.name, p, d.level, minimum(mt), minimum(ut), minimum(ut) / minimum(mt),
                    median_time(mt), median_time(ut), median_time(ut) / median_time(mt), max_solution,
                    max_objective, max_kkt, max_stationarity, converged, valid), ','))
                flush(summary)
                @printf "%s p=%d level=%g U/M median=%.3f best=%.3f KKT=%.2e stationarity=%.2e solution_diff=%.2e valid=%s\n" d.name p d.level median_time(ut)/median_time(mt) minimum(ut)/minimum(mt) max_kkt max_stationarity max_solution valid
            end
        end
    end
    println("raw: ", OUT)
    println("summary: ", summary_path)
    all_ok || error("at least one cache comparison failed validation; retain the output and inspect its diagnostics")
    println("KRON_CACHE_BENCHMARK_VALIDATED")
end

main()
