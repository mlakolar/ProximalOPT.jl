# Optional reference. Run in an environment containing MendelIHT, after
# legacy.jl --prepare has written the common centered regression data.
# API verified against https://github.com/OpenMendel/MendelIHT.jl/blob/master/src/fit.jl
include("common.jl")

function main()
    data = deserialize(joinpath(ROOT, "regression.jls"))
    open(joinpath(ROOT, "mendel_regression.tsv"), "w") do io
        record(io, ("solver", "round", "seconds", "iterations", "objective", "support", "support_matches_truth", "status", "detail"))
        if Base.find_package("MendelIHT") === nothing
            record(io, ("MendelIHT", 0, NaN, -1, Inf, "", false, "unavailable", "MendelIHT is not installed in this environment"))
            return
        end
        try
            @eval import MendelIHT
        catch err
            record(io, ("MendelIHT", 0, NaN, -1, Inf, "", false, "unavailable", sprint(showerror, err)))
            return
        end
        # Centered X and y make the intercept's least-squares optimum zero.
        # Keep MendelIHT's documented intercept and record its effect through
        # the independently evaluated residual when the field is exposed.
        fit(debias, max_iter) = Base.invokelatest(MendelIHT.fit_iht, data.y, data.X; k = data.s,
            debias, verbose = false, init_beta = false, max_iter, tol = 1e-8, max_step = 3)
        fit(false, 5)                # compile, excluded from times
        for round in 1:ROUNDS, debias in (false, true)
            try
                elapsed = @elapsed result = fit(debias, MAXITER)
                beta = Vector(result.beta)
                intercept = hasproperty(result, :c) ? first(result.c) : 0.0
                F = sum(abs2, data.y .- data.X * beta .- intercept) / (2length(data.y))
                support = findall(!iszero, beta)
                iterations = hasproperty(result, :iter) ? result.iter :
                             hasproperty(result, :iterations) ? result.iterations : -1
                record(io, ("MendelIHT_debias_$debias", round, elapsed, iterations, F, join(support, ';'),
                    support == data.support, "returned", "version=$(pkgversion(MendelIHT)); intercept=$intercept"))
            catch err
                record(io, ("MendelIHT_debias_$debias", round, NaN, -1, Inf, "", false, "error", sprint(showerror, err)))
            end
        end
    end
    println("MENDEL_COMPARISON_COMPLETE")
end
main()
