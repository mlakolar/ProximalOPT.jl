# Tables from the results of run.jl (results/*.csv, or the folder in PROXALG_OUT).
#
#     julia --project=benchmark/proxalg benchmark/proxalg/summary.jl

using Printf

const DIR = get(ENV, "PROXALG_OUT", joinpath(@__DIR__, "results"))
const TARGETS = (1e-4, 1e-6, 1e-8)

function readcsv(file)
  lines = readlines(joinpath(DIR, file))
  header = split(lines[1], ',')
  [Dict(zip(header, split(l, ','; limit = length(header)))) for l in lines[2:end] if !isempty(l)]
end
num(s) = (x = tryparse(Float64, s); x === nothing ? NaN : x)
fmt_t(t) = isfinite(t) ? (t < 1e-3 ? @sprintf("%.2g ms", 1e3t) : t < 1 ? @sprintf("%.3g ms", 1e3t) : @sprintf("%.3g s", t)) : "—"
fmt_b(b) = isfinite(b) ? (b < 1e3 ? @sprintf("%.0f B", b) : b < 1e6 ? @sprintf("%.1f KB", b / 1e3) : @sprintf("%.1f MB", b / 1e6)) : "—"
rel(F, Fs) = isfinite(F) ? max((F - Fs) / max(abs(Fs), 1.0), 0.0) : Inf

function main()
  sweep, periter = readcsv("sweep.csv"), readcsv("per_iteration.csv")
  probe = isfile(joinpath(DIR, "inf_probe.csv")) ? readcsv("inf_probe.csv") : []
  problems = unique(r["problem"] for r in sweep)
  for P in problems
    rows = filter(r -> r["problem"] == P, sweep)
    solvers = unique((r["solver"], r["package"]) for r in rows)
    Fs = [num(r["F"]) for r in rows if r["status"] == "ok" && isfinite(num(r["F"]))]
    Fref = num(rows[1]["Fstar"])
    convex = isfinite(Fref)
    Fstar = convex ? min(Fref, minimum(Fs; init = Inf)) : minimum(Fs; init = Inf)
    println("\n### ", P, convex ? "" : " (nonconvex)")
    convex && Fstar < Fref - 1e-9 * max(abs(Fref), 1.0) &&
      @printf("Note: a run beat the reference by %.1e (relative).\n", (Fref - Fstar) / max(abs(Fref), 1.0))
    if convex
      println("\n| solver | package | ", join(("time to $(t)" for t in TARGETS), " | "),
              " | best rel. suboptimality | bytes/iteration | time/iteration | note |")
      println("|---|---|", repeat("---:|", length(TARGETS) + 3), "---|")
    else
      println("\n| solver | package | best F − F_best (rel.) | time to F_best + 1e-6 | time to its own best | iterations | bytes/iteration | time/iteration | note |")
      println("|---|---|---:|---:|---:|---:|---:|---:|---|")
    end
    for (s, pkg) in solvers
      rs = filter(r -> r["solver"] == s && r["package"] == pkg, rows)
      ok = filter(r -> r["status"] == "ok", rs)
      errs = filter(r -> r["status"] != "ok", rs)
      note = isempty(errs) ? "" : "error: " * first(errs)["status"]
      any(r -> !isfinite(num(r["F"])), ok) && (note = strip(note * " F = NaN or Inf in some runs"))
      pi = filter(r -> r["solver"] == s && r["package"] == pkg && r["problem"] == P, periter)
      b = isempty(pi) ? NaN : num(pi[1]["bytes"])
      ti = isempty(pi) ? NaN : num(pi[1]["time"])
      if convex
        ts = [minimum((num(r["time"]) for r in ok if rel(num(r["F"]), Fstar) <= t); init = Inf) for t in TARGETS]
        best = minimum((rel(num(r["F"]), Fstar) for r in ok); init = Inf)
        @printf("| %s | %s | %s | %.1e | %s | %s | %s |\n", s, pkg, join(fmt_t.(ts), " | "), best, fmt_b(b), fmt_t(ti), note)
      else
        gaps = [rel(num(r["F"]), Fstar) for r in ok]
        best = minimum(gaps; init = Inf)
        t6 = minimum((num(r["time"]) for (r, g) in zip(ok, gaps) if g <= 1e-6); init = Inf)
        town = minimum((num(r["time"]) for (r, g) in zip(ok, gaps) if g <= best + 1e-8); init = Inf)
        it = isempty(ok) ? -1 : parse(Int, ok[argmin(gaps)]["iterations"])
        @printf("| %s | %s | %.1e | %s | %s | %d | %s | %s | %s |\n", s, pkg, best, fmt_t(t6), fmt_t(town), it,
                fmt_b(b), fmt_t(ti), note)
      end
    end
  end
  if !isempty(probe)
    println("\n### Graphical lasso from a first step that leaves the cone")
    println("\n| p | solver | package | initial step | F − F* (rel.) | iterations | time | note |")
    println("|---:|---|---|---:|---:|---:|---:|---|")
    for r in probe
      F, Fs = num(r["F"]), num(r["Fstar"])
      @printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n", r["p"], r["solver"], r["package"], r["gamma0"],
              isfinite(F) ? @sprintf("%.1e", rel(F, Fs)) : string(F), r["iterations"], fmt_t(num(r["time"])),
              r["status"] == "ok" ? "" : r["status"])
    end
  end
end

main()
