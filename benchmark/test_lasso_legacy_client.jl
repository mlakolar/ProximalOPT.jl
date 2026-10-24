# Stdlib-only lifecycle checks; run on a compute node with either supported Julia.
using Serialization, Test
include(joinpath(@__DIR__, "lasso_common.jl"))
include(joinpath(@__DIR__, "lasso_legacy_client.jl"))

function test_worker(body)
  loadpath = join(("@", "@stdlib"), Sys.iswindows() ? ';' : ':')
  cmd = addenv(`$(Base.julia_cmd()) --startup-file=no -e $body`, "JULIA_LOAD_PATH" => loadpath)
  open(cmd, "r+")
end

function caught(f)
  try
    f()
    nothing
  catch err
    err
  end
end

function check_reaped(worker)
  @test LEGACY_WORKER[] === nothing
  @test isempty(LEGACY_PROBLEMS)
  @test !isopen(worker.in)
  @test !isopen(worker.out)
  @test process_exited(worker)
end

const READ_STOP = """
while !eof(stdin)
  request = deserialize(stdin)
  request.operation === :stop && break
end
"""

@testset "Historical worker lifecycle" begin
  @testset "EOF during handshake" begin
    worker = test_worker("exit(3)")
    err = caught(() -> initialize_legacy_worker!(worker))
    @test err isa EOFError
    check_reaped(worker)
  end

  @testset "Rejected handshake preserves its error" begin
    worker = test_worker("""
      using Serialization
      serialize(stdout, (ok = false, error = "intentional handshake failure"))
      flush(stdout)
      $READ_STOP
      """)
    err = caught(() -> initialize_legacy_worker!(worker))
    @test err isa ErrorException
    @test occursin("intentional handshake failure", sprint(showerror, err))
    check_reaped(worker)
  end

  @testset "Normal shutdown" begin
    worker = test_worker("""
      using Serialization
      serialize(stdout, (ok = true, environment = "test worker"))
      flush(stdout)
      $READ_STOP
      """)
    @test initialize_legacy_worker!(worker) === worker
    push!(LEGACY_PROBLEMS, UInt(1))
    @test close_legacy_worker!() === nothing
    check_reaped(worker)
  end

  @testset "Stop-write failure still reaps the child" begin
    mktempdir() do dir
      release = joinpath(dir, "release")
      worker = test_worker("""
        using Serialization
        serialize(stdout, (ok = true, environment = "test worker"))
        flush(stdout)
        while !isfile($(repr(release)))
          sleep(0.001)
        end
        """)
      initialize_legacy_worker!(worker)
      close(worker.in)
      # Keep the child alive until cleanup closes its output pipe. The bounded
      # fallback also lets a failing implementation exit without leaking it.
      releaser = @async begin
        for _ in 1:1000
          isopen(worker.out) || break
          sleep(0.001)
        end
        touch(release)
      end
      err = caught(close_legacy_worker!)
      @test err !== nothing
      check_reaped(worker)
      wait(releaser)
    end
  end
end
println("LASSO_LEGACY_LIFECYCLE_PASSED")
