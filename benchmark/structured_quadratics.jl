# Run inside a Discovery Slurm allocation with this checkout active:
# julia --project=/path/to/environment benchmark/structured_quadratics.jl
module StructuredQuadraticBenchmark

using LinearAlgebra, SparseArrays, Printf
using ProximalBase, ProximalSolvers
const PS = ProximalSolvers
BLAS.set_num_threads(1)

# The previous constructor materialized every non-Matrix Hessian. The new
# dense model uses the same coordinate formulas, now through an inline helper.
legacy_model(f) = PS.QuadModel(Matrix(f.A), f.b, f.c, 1)

function legacy_dense_column!(u, A, k, h)
  @inbounds @simd for i in eachindex(u)
    u[i] += A[i, k] * h
  end
  u
end

function measure(f::F, setup::S; samples = 31) where {F, S}
  setup(); f()
  setup(); bytes = @allocated f()
  times = Vector{Float64}(undef, samples)
  for i in eachindex(times)
    setup()
    times[i] = @elapsed f()
  end
  sort!(times)
  (ns = 1e9 * times[(samples + 1) ÷ 2], bytes = bytes)
end

function sweep!(m, x, g)
  for k in 1:ncoordinates(x)
    PS.coordinate_step!(m, g, x, k)
  end
  x
end

function group_sweep!(m, x, g)
  for k in 1:ncoordinates(x)
    PS._groupstep!(m, g, x, k, false)
  end
  x
end

function reset!(m, x)
  fill!(x, zero(eltype(x)))
  PS._initialize!(m, x, false)
end

function row(name, operation, p, a, b, dense_size, structured_size)
  @printf "%s,%s,%d,%.0f,%.0f,%d,%d,%d,%d\n" name operation p a.ns b.ns a.bytes b.bytes dense_size structured_size
end

function main()
  println("case,operation,p,dense_ns,structured_ns,dense_allocated_bytes,structured_allocated_bytes,dense_model_bytes,structured_model_bytes")
  for p in (500, 2000), diagonal in (false, true)
    name = diagonal ? "diagonal" : "sparse_band"
    A = diagonal ? Diagonal(fill(2.0, p)) :
        spdiagm(-1 => fill(-0.2, p - 1), 0 => fill(2.0, p), 1 => fill(-0.2, p - 1))
    b = fill(-0.5, p)
    f = QuadraticFunction(A, b)
    x, xd = SparseIterate(p), SparseIterate(p)
    m, md = cdmodel(f, x), legacy_model(f)
    dense_size, structured_size = Base.summarysize(md), Base.summarysize(m)
    # Nine samples bound the repeated dense setup allocations at the larger p.
    a = measure(() -> legacy_model(f), () -> nothing; samples = 9)
    c = measure(() -> cdmodel(f, x), () -> nothing; samples = 9)
    row(name, "model_setup", p, a, c, dense_size, structured_size)

    g = NormL1(0.1)
    PS.initialize!(m, x); PS.initialize!(md, xd)
    sweep!(m, x, g); sweep!(md, xd, g)
    @assert Vector(x) ≈ Vector(xd)
    a = measure(() -> sweep!(md, xd, g), () -> reset!(md, xd))
    c = measure(() -> sweep!(m, x, g), () -> reset!(m, x))
    row(name, "coordinate_sweep", p, a, c, dense_size, structured_size)

    groups = [j:min(j + 9, p) for j in 1:10:p]
    xa, xad = AtomIterate(p, groups), AtomIterate(p, groups)
    ma, mad = cdmodel(f, xa), legacy_model(f)
    gg = GroupNormL2(0.1, groups)
    PS.initialize!(ma, xa); PS.initialize!(mad, xad)
    group_sweep!(ma, xa, gg); group_sweep!(mad, xad, gg)
    @assert Vector(xa) ≈ Vector(xad)
    a = measure(() -> group_sweep!(mad, xad, gg), () -> reset!(mad, xad))
    c = measure(() -> group_sweep!(ma, xa, gg), () -> reset!(ma, xa))
    row(name, "group_sweep", p, a, c, Base.summarysize(mad), Base.summarysize(ma))

    if !diagonal
      u = zeros(p)
      a = measure(() -> legacy_dense_column!(u, md.A, 1, 0.1), () -> fill!(u, 0.0))
      c = measure(() -> PS._quad_axpy!(u, md.A, 1, 0.1), () -> fill!(u, 0.0))
      row("dense", "column_update", p, a, c, dense_size, dense_size)
    end
  end
end

main()
end
