# Run from the package checkout after instantiating its environment:
# julia --project=. benchmark/path_snapshots.jl
# The legacy methods below reproduce the snapshots before this change.
using Printf, Random, SparseArrays, ProximalBase, ProximalSolvers
const PS = ProximalSolvers

function measure(f::F) where {F}
  f()                         # compile before measuring
  bytes = @allocated f()
  times = [@elapsed f() for _ in 1:50]
  sort!(times)
  (time = 1e9 * (times[25] + times[26]) / 2, memory = bytes)
end

function old_snapshot(x::SparseIterate{T, 1}) where {T}
  ind, val = Int[], T[]
  for k in activecoordinates(x)
    v = x[k]
    iszero(v) && continue
    push!(ind, k); push!(val, v)
  end
  p = sortperm(ind)
  sparsevec(ind[p], val[p], length(x))
end
old_snapshot(x::AtomIterate) = sparsevec(Vector(x))
old_snapshot(x::SymmetricSparseIterate) = sparse(Matrix(x))
old_snapshot(x::SparseIterate{T, 2}) where {T} = sparse(Matrix(x))

function snapshots()
  cases = (("vector", SparseIterate(100_000)),
           ("groups", AtomIterate(100_000, [k:k+9 for k in 1:10:100_000])),
           ("matrix", SparseIterate(800, 1200)),
           ("symmetric", SymmetricSparseIterate(1000)))
  println("case,old_ns,new_ns,old_bytes,new_bytes")
  for (name, x) in cases
    for k in 20:-1:1
      x[17k] = isodd(k) ? Float64(k) : -Float64(k)
    end
    @assert old_snapshot(x) == PS._snapshot(x)
    a = measure(() -> old_snapshot(x))
    b = measure(() -> PS._snapshot(x))
    @printf "%s,%.0f,%.0f,%d,%d\n" name a.time b.time a.memory b.memory
  end
end

old_copyback!(dest::Vector, src) = copyto!(dest, Vector(src))
old_copyback!(dest::Matrix, src) = copyto!(dest, Matrix(src))

function copyback()
  println("copyback,old_ns,new_ns,old_bytes,new_bytes")
  for src in (SparseIterate(100_000), SymmetricSparseIterate(1000))
    src[10] = 1.0
    dest = zeros(size(src))
    a = measure(() -> old_copyback!(dest, src))
    b = measure(() -> PS._copy_dense!(dest, src))
    @printf "%s,%.0f,%.0f,%d,%d\n" nameof(typeof(src)) a.time b.time a.memory b.memory
  end
end

function density_probe()
  println("density_case,requested_density,actual_density,old_ns,new_ns,old_bytes,new_bytes")
  for (name, make) in (("matrix", () -> SparseIterate(200, 300)),
                       ("symmetric", () -> SymmetricSparseIterate(300)),
                       ("groups", () -> AtomIterate(60_000, [k:k+9 for k in 1:10:60_000])))
    template = make()
    order = randperm(Xoshiro(981), ncoordinates(template))
    for density in (0.01, 0.10, 0.50, 1.0)
      x = make()
      for k in view(order, 1:round(Int, density * length(order)))
        if x isa AtomIterate
          for j in x.groups[k]
            x[j] = isodd(j) ? 1.0 : -1.0
          end
        else
          coordinates(x)[k] = isodd(k) ? 1.0 : -1.0
        end
      end
      reference = old_snapshot(x)
      @assert reference == PS._snapshot(x)
      actual = nnz(reference) / length(x)
      GC.gc()
      a = measure(() -> old_snapshot(x))
      GC.gc()
      b = measure(() -> PS._snapshot(x))
      @printf "%s,%.2f,%.6f,%.0f,%.0f,%d,%d\n" name density actual a.time b.time a.memory b.memory
    end
  end
end

snapshots()
copyback()
density_probe()
