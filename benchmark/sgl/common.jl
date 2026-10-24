# reading a problem written by problems.jl

function readproblem(d)
  meta = Dict(Symbol(first(split(kv, "="))) => parse(Int, last(split(kv, "="))) for kv in split(readline(joinpath(d, "meta.txt"))))
  n, p, gsize = meta[:n], meta[:p], meta[:gsize]
  X = reshape(reinterpret(Float64, read(joinpath(d, "X.bin"))), n, p) |> Matrix
  y = Vector(reinterpret(Float64, read(joinpath(d, "y.bin"))))
  groups = [i:i+gsize-1 for i in 1:gsize:p]
  X, y, groups
end
