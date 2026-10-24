# Reproducible, interleaved graphical-lasso paths across Julia and R.
# Run this whole script under Slurm on Discovery, never on a login node.
# R package setup is a separate recorded step inside the same allocation.
#
# julia --project=benchmark/quic benchmark/quic/cross_language.jl \
#   --root /absolute/run/results/cross_language --sizes 100,300,600,2000 \
#   --lambda 0.2,0.12,0.08 --rounds 2 --rscript /absolute/path/Rscript
#
# All five solvers minimize tr(S Θ) - logdet(Θ) + λ Σ_{i≠j}|Θ[i,j]|.
# Every returned precision is validated in Julia with an objective computed
# directly from its Cholesky factor and the common graphical-lasso dual gap.
# R startup, file I/O and validation are excluded from reported solve times.

using DelimitedFiles, LinearAlgebra, Random, Printf
using ProximalBase, ProximalSolvers
import GraphicalLassoQUIC

function option(name, default)
    k = findfirst(==(name), ARGS)
    k === nothing && return default
    k < length(ARGS) || error("$name needs an argument")
    ARGS[k + 1]
end

const ROOT = abspath(option("--root", joinpath(@__DIR__, "cross_language_results")))
const SIZES = parse.(Int, split(option("--sizes", "100,300,600,2000"), ','))
const LAMBDA = parse.(Float64, split(option("--lambda", "0.2,0.12,0.08"), ','))
const ROUNDS = parse(Int, option("--rounds", "2"))
const TARGET = parse(Float64, option("--target-gap", "1e-4"))
const REFERENCE_TOL = parse(Float64, option("--reference-tol", "1e-4"))
const GLASSO_THR = parse(Float64, option("--glasso-thr", "1e-6"))
const MAXITER = parse(Int, option("--max-iter", "100"))
const FIT_LIMIT = parse(Float64, option("--fit-limit", "120"))
const RSCRIPT = option("--rscript", "Rscript")
const RFILE = joinpath(@__DIR__, "cross_language.R")
const SOLVERS = ("native_quic", "admm", "graphicalquic", "R_QUIC", "R_glasso")

clean(s) = replace(string(s), '\t' => ' ', '\r' => ' ', '\n' => ' ')
write_row(io, values) = (println(io, join(clean.(values), '\t')); flush(io))
theta_file(p, round, solver, k) = "theta_p$(p)_r$(round)_$(solver)_k$(k).bin"
metadata_file(p, round, solver) = joinpath(ROOT, "p$(p)_r$(round)_$(solver).tsv")

function read_matrix(path, p)
    filesize(path) == 8p * p || error("wrong matrix file size for $path")
    read!(path, Matrix{Float64}(undef, p, p))
end

# Diagonally dominant sparse precision; n=2p Gaussian observations. Both
# languages use the identical exported S. No standardization, intercept or
# implicit n-scaling is applied to the common likelihood objective.
function covariance(p)
    omega = Matrix(1.4I, p, p)
    for k in 1:p-1
        omega[k, k + 1] = omega[k + 1, k] = 0.3
    end
    for k in 1:p-4
        omega[k, k + 4] = omega[k + 4, k] = 0.1
    end
    Z = randn(Xoshiro(2200 + p), 2p, p) / cholesky(Symmetric(omega)).U'
    Matrix(Symmetric(Z' * Z / (2p)))
end

function native_path(S, p, round, solver; save = true)
    rows = Any[]
    X = Matrix(1.0I, p, p)
    admm_previous = nothing
    for (k, lambda) in enumerate(LAMBDA)
        weights = fill(lambda, p, p)
        weights[diagind(weights)] .= 0
        f, g = LogDetLoss(S), NormL1(1.0, weights)
        iterations, status, detail = -1, "returned", ""
        theta = nothing
        GC.gc()
        seconds = @elapsed try
            if solver == "native_quic"
                # Reusing X warm starts the primal path. Its inverse and all
                # other setup required by solve! stay in the timed call.
                r = solve!(X, f, g, ProxNewton(; maxiter = MAXITER, tol = TARGET,
                    inner = CDSolver(; tol = 1e-8, rng = Xoshiro(2200 + p + round))))
                theta = copy(X)
                iterations = r.passes
                detail = "passes; converged=$(r.converged); warm=$(k > 1)"
            elseif solver == "admm"
                opt = ProximalOptions(; maxiter = 5000, time_limit = FIT_LIMIT,
                                      gaptol = TARGET, gapevery = 10)
                r = admm_previous === nothing ?
                    solve!(X, f, g, ADMMSolver(; adaptive = true), opt) :
                    solve!(X, f, g, ADMMSolver(; adaptive = true, ρ = admm_previous.ρ), opt;
                           u = admm_previous.u)
                admm_previous = r
                theta = copy(X)
                iterations = r.iterations
                detail = "iterations; converged=$(r.converged); soft_limit=$(FIT_LIMIT)s; warm=$(k > 1)"
            else
                # This package returns Θ, not its inverse. Computing Winit
                # for the warm start is therefore part of its timed work.
                X = k == 1 ? GraphicalLassoQUIC.QUIC(S, weights;
                    tol = REFERENCE_TOL, maxIter = MAXITER, rng = Xoshiro(2200 + p + round)) :
                    GraphicalLassoQUIC.QUIC(S, weights; tol = REFERENCE_TOL, maxIter = MAXITER,
                        Xinit = X, Winit = inv(Symmetric(X)), rng = Xoshiro(2200 + p + round))
                theta = copy(X)
                detail = "iteration count not returned; warm=$(k > 1)"
            end
        catch err
            status, detail = "error", sprint(showerror, err)
        end
        file = ""
        if theta !== nothing
            file = theta_file(p, round, solver, k)
            save && write(joinpath(ROOT, file), theta)
        end
        push!(rows, (p, round, k, lambda, solver, seconds, iterations, status, file, detail))
    end
    if save
        open(metadata_file(p, round, solver), "w") do io
            write_row(io, ("p", "round", "level", "lambda", "solver", "seconds", "iterations", "status", "theta_file", "detail"))
            for row in rows
                write_row(io, row)
            end
        end
    end
    rows
end

function unavailable(p, round, solver, detail)
    open(metadata_file(p, round, solver), "w") do io
        write_row(io, ("p", "round", "level", "lambda", "solver", "seconds", "iterations", "status", "theta_file", "detail"))
        for (k, lambda) in enumerate(LAMBDA)
            write_row(io, (p, round, k, lambda, solver, NaN, -1, "unavailable", "", detail))
        end
    end
end

function r_path(p, round, solver)
    if Sys.which(RSCRIPT) === nothing && !isfile(RSCRIPT)
        unavailable(p, round, solver, "Rscript not found: $RSCRIPT")
        return
    end
    command = Cmd([RSCRIPT, "--vanilla", RFILE, "--root", ROOT, "--p", string(p),
                   "--round", string(round), "--solver", solver,
                   "--reference-tol", string(REFERENCE_TOL), "--glasso-thr", string(GLASSO_THR),
                   "--max-iter", string(MAXITER), "--fit-limit", string(FIT_LIMIT)])
    process = run(ignorestatus(command); wait = false)
    limit = length(LAMBDA) * FIT_LIMIT + 60
    state = timedwait(() -> process_exited(process), limit; pollint = 0.25)
    if state == :timed_out
        kill(process)
        wait(process)
        # Keep any rows already flushed by the R process; validation adds
        # explicit unavailable rows for missing levels rather than guessing.
        isfile(metadata_file(p, round, solver)) || unavailable(p, round, solver, "R child timeout after $(limit)s")
        @warn "R path timed out" p round solver limit
    elseif !success(process)
        isfile(metadata_file(p, round, solver)) || unavailable(p, round, solver, "R child failed, exit $(process.exitcode)")
        @warn "R path failed" p round solver exitcode = process.exitcode
    end
end

function validate(S, theta, lambda)
    asymmetry = norm(theta - theta') / max(1.0, norm(theta))
    all(isfinite, theta) || return (NaN, NaN, NaN, asymmetry, false)
    symmetric = Matrix(Symmetric((theta + theta') / 2))
    factor = cholesky(Symmetric(symmetric); check = false)
    issuccess(factor) || return (Inf, Inf, Inf, asymmetry, false)
    weights = fill(lambda, size(S))
    weights[diagind(weights)] .= 0
    objective = dot(S, symmetric) - 2sum(log, diag(factor.U)) + sum(weights .* abs.(symmetric))
    gap = duality_gap(LogDetLoss(S), NormL1(1.0, weights), symmetric)
    relgap = max(gap.gap, 0.0) / max(abs(objective), floatmin(Float64))
    consistent = abs(objective - gap.primal) <= 1e-9 * max(1.0, abs(objective))
    valid = asymmetry <= 1e-8 && consistent && isfinite(relgap) &&
            gap.gap >= -1e-8 * max(1.0, abs(objective)) && relgap <= 1.05TARGET
    objective, gap.gap, relgap, asymmetry, valid
end

function main()
    ENDIAN_BOM == 0x04030201 || error("binary exchange requires a little-endian host")
    all(>=(5), SIZES) || error("all sizes must be at least 5")
    ROUNDS >= 1 && MAXITER >= 1 && FIT_LIMIT > 0 && TARGET > 0 || error("invalid run limits")
    all(>(0), LAMBDA) && issorted(LAMBDA; rev = true) || error("lambda must be positive and descending")
    mkpath(ROOT)
    BLAS.set_num_threads(1)
    writedlm(joinpath(ROOT, "lambda.txt"), LAMBDA)
    open(joinpath(ROOT, "configuration.txt"), "w") do io
        println(io, "Julia=", VERSION, " GraphicalLassoQUIC=", pkgversion(GraphicalLassoQUIC))
        println(io, "sizes=", SIZES, " lambda=", LAMBDA, " rounds=", ROUNDS)
        println(io, "target_gap=", TARGET, " reference_tol=", REFERENCE_TOL, " glasso_thr=", GLASSO_THR)
        println(io, "max_newton_or_glasso_iterations=", MAXITER, " ADMM_maxiter=5000 ADMM_time_limit=", FIT_LIMIT)
        println(io, "Rscript=", RSCRIPT, " BLAS_threads=", BLAS.get_num_threads())
        println(io, "precision_binary=Float64_little_endian_column_major")
    end
    if !("--validate-only" in ARGS)
        # Compile native paths on a separate small covariance before timing.
        for solver in SOLVERS[1:3]
            native_path(covariance(8), 8, 1, solver; save = false)
        end
        for p in SIZES
            S = covariance(p)
            write(joinpath(ROOT, "S_p$(p).bin"), S)
            for round in 1:ROUNDS
                order = isodd(round) ? SOLVERS : reverse(SOLVERS)
                for solver in order
                    println("RUN p=", p, " round=", round, " solver=", solver)
                    startswith(solver, "R_") ? r_path(p, round, solver) : native_path(S, p, round, solver)
                end
            end
        end
    end
    output = joinpath(ROOT, "validated.tsv")
    validated, unavailable_count, invalid = 0, 0, 0
    open(output, "w") do io
        write_row(io, ("p", "round", "level", "lambda", "solver", "seconds", "iterations", "status", "objective", "duality_gap", "relative_gap", "relative_asymmetry", "validated", "detail"))
        for p in SIZES
            S = read_matrix(joinpath(ROOT, "S_p$(p).bin"), p)
            for round in 1:ROUNDS, solver in SOLVERS
                file = metadata_file(p, round, solver)
                data = isfile(file) ? readdlm(file, '\t', Any; header = true)[1] : Matrix{Any}(undef, 0, 10)
                for (k, lambda) in enumerate(LAMBDA)
                    matches = findall(i -> Int(data[i, 3]) == k, axes(data, 1))
                    if isempty(matches)
                        unavailable_count += 1
                        write_row(io, (p, round, k, lambda, solver, NaN, -1, "unavailable", NaN, NaN, NaN, NaN, false, "missing result; inspect child log"))
                        continue
                    end
                    row = data[first(matches), :]
                    status, detail = string(row[8]), string(row[10])
                    objective, gap, relgap, asymmetry, valid = NaN, NaN, NaN, NaN, false
                    if status == "returned"
                        try
                            theta = read_matrix(joinpath(ROOT, string(row[9])), p)
                            objective, gap, relgap, asymmetry, valid = validate(S, theta, lambda)
                        catch err
                            status, detail = "validation_error", detail * "; " * sprint(showerror, err)
                        end
                        valid ? (validated += 1) : (invalid += 1)
                    else
                        unavailable_count += 1
                    end
                    write_row(io, (p, round, k, lambda, solver, row[6], row[7], status,
                                   objective, gap, relgap, asymmetry, valid, detail))
                end
            end
        end
    end
    println("CROSS_LANGUAGE_COMPLETED validated=", validated, " invalid=", invalid,
            " unavailable_or_failed=", unavailable_count, " results=", output)
    invalid == 0 && unavailable_count == 0 || error("cross-language comparison has invalid or unavailable results; inspect validated.tsv")
    println("CROSS_LANGUAGE_VALIDATED")
end

main()
