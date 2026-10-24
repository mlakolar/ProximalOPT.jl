# The historical baseline needs ProximalBase 0.4, whereas the solver under test
# needs 0.5. Keep one baseline process, send each actual design once, and time
# its calls inside that process so IPC and startup are outside the measurement.
struct LegacyRun
  operation::Symbol
end

const LEGACY_WORKER = Ref{Union{Nothing, Base.Process}}(nothing)
const LEGACY_PROBLEMS = Set{UInt}()

function legacy_worker()
  worker = LEGACY_WORKER[]
  worker === nothing || return worker
  env = get(ENV, "PROXIMAL_LEGACY_CD_PROJECT", joinpath(@__DIR__, "legacy_cd"))
  script = joinpath(@__DIR__, "legacy_cd", "worker.jl")
  # A caller's custom LOAD_PATH must not leak current dependencies into the
  # historical environment. The depot can still cache both package versions.
  loadpath = join(("@", "@stdlib"), Sys.iswindows() ? ';' : ':')
  cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env $script`,
               "JULIA_LOAD_PATH" => loadpath)
  worker = open(cmd, "r+")
  hello = deserialize(worker)
  hello.ok || error(hello.error)
  println("Historical worker: ", hello.environment)
  LEGACY_WORKER[] = worker
  worker
end

function close_legacy_worker!()
  worker = LEGACY_WORKER[]
  worker === nothing && return nothing
  LEGACY_WORKER[] = nothing
  empty!(LEGACY_PROBLEMS)
  if process_running(worker)
    serialize(worker, (operation = :stop,))
    flush(worker)
  end
  close(worker)
  wait(worker)
  success(worker) || error("Historical coordinate-descent worker failed.")
  nothing
end
atexit(close_legacy_worker!)

function legacy_request(run::LegacyRun, P::LassoProblem, args...; timing = nothing)
  worker = legacy_worker()
  key = objectid(P)
  problem = key in LEGACY_PROBLEMS ? nothing : P
  serialize(worker, (; operation = run.operation, key, problem, args, timing))
  flush(worker)
  response = deserialize(worker)
  response.ok || error(response.error)
  push!(LEGACY_PROBLEMS, key)
  response
end

(run::LegacyRun)(P::LassoProblem, args...) = legacy_request(run, P, args...).result

timed_solver(run, P, args...; kwargs...) = timed(() -> run(P, args...); kwargs...)
function timed_solver(run::LegacyRun, P, args...; budget = 1.0, minreps = 3, maxreps = 7)
  response = legacy_request(run, P, args...; timing = (; budget, minreps, maxreps))
  response.result, response.time
end
