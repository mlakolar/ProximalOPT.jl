# Summarize existing solutions inside the same Discovery Slurm allocation.
# This never refits a model. Support disagreements are retained, not repaired.
include("common.jl")
cases = deserialize(joinpath(ROOT, "cases.jls"))
legacy = deserialize(joinpath(ROOT, "legacy_solutions.jls"))
native = deserialize(joinpath(ROOT, "native_solutions.jls"))
open(joinpath(ROOT, "native_vs_legacy.tsv"), "w") do io
    record(io, ("problem", "p", "solver", "round", "native_objective", "best_fixed_step_objective",
                "native_minus_legacy_relative", "same_support", "native_support", "legacy_support", "native_no_worse", "detail"))
    for row in native
        matches = filter(x -> x.problem == row.problem && x.p == row.p && isfinite(x.objective), legacy)
        isempty(matches) && continue
        reference = matches[argmin(getproperty.(matches, :objective))]
        d = first(filter(x -> x.name == row.problem && x.p == row.p, cases))
        a, b = support_string(row.S, d.diagonal), support_string(reference.S, d.diagonal)
        delta = (row.objective - reference.objective) / max(1.0, abs(reference.objective))
        record(io, (row.problem, row.p, row.method, row.round, row.objective, reference.objective,
                    delta, a == b, a, b, delta <= 1e-8,
                    "legacy best over recorded fixed-step grid; nonconvex local solutions may differ"))
    end
end
println("NONCONVEX_COMPARISON_SUMMARIZED")
