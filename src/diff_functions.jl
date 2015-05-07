
##########################################################
#
#  Create DifferentiableFunction
#
##########################################################

abstract DifferentiableFunction

gradient{T<:FloatingPoint}(f::DifferentiableFunction, x::StridedVector{T}) = gradient!(f, similar(x), x)

################ creates L2Loss =>  f(x) = ||x - y||^2 / 2

# N = 1 --> y
# N = 0 --> y = 0
immutable L2Loss{T<:FloatingPoint, N} <: DifferentiableFunction
  y::StridedArray{T}
end

L2Loss{T<:FloatingPoint}(y::StridedArray{T}) = L2Loss{T, 1}(y)
L2Loss() = L2Loss{Float64, 0}(Float64[])


function value{T<:FloatingPoint, N}(f::L2Loss{T, N}, x::StridedArray{T})
  s = zero(T)
  if N == 1
    y = f.y
    @assert size(y) == size(x)
    @inbounds for i in eachindex(x)
      s += (x[i]-y[i])^2.
    end
  else
    s = sumabs2(x)
  end
  s / 2.
end


function value{T<:FloatingPoint, N}(f::L2Loss{T, N}, x::StridedArray{T}, activeset::ActiveSet)
  s = zero(T)
  if N == 1
    y = f.y
    @assert size(y) == size(x)
  end
  indexes = activeset.indexes
  @inbounds for i=1:activeset.numActive
    ind = indexes[i]
    if N == 1
      s += (x[ind]-y[ind])^2.
    else
      s += x[ind]^2.
    end
  end
  s / 2.
end

gradient!{T<:FloatingPoint}(f::L2Loss{T, 0}, hat_x::StridedArray{T}, x::StridedArray{T}) = copy!(hat_x, x)
function gradient!{T<:FloatingPoint}(f::L2Loss{T, 1}, hat_x::StridedArray{T}, x::StridedArray{T})
  y = f.y
  @assert size(y) == size(x)
  @inbounds for i in eachindex(x)
    hat_x[i] = x[i]-y[i]
  end
  hat_x
end
function gradient!{T<:FloatingPoint, N}(f::L2Loss{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::ActiveSet)
  if N == 1
    y = f.y
    @assert size(y) == size(x)
  end
  indexes = activeset.indexes
  @inbounds for rr=1:activeset.numActive
    t = indexes[rr]
    hat_x[t] = N == 1 ? x[t] - y[t] : x[t]
  end
  hat_x
end

function value_and_gradient!{T<:FloatingPoint, N}(f::L2Loss{T, N}, hat_x::StridedArray{T}, x::StridedArray{T})
  if N == 1
    y = f.y
    @assert size(y) == size(x)
    @inbounds for i in eachindex(x)
      hat_x[i] = x[i] - y[i]
    end
  else
    copy!(hat_x, x)
  end
  sumabs2(hat_x) / 2.
end

function value_and_gradient!{T<:FloatingPoint, N}(f::L2Loss{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::ActiveSet)
  s = zero(T)
  if N == 1
    y = f.y
    @assert size(y) == size(x)
  end
  indexes = activeset.indexes
  @inbounds for rr=1:activeset.numActive
    t = indexes[rr]
    s += N == 1 ? (x[t]-y[t])^2. : x[t]^2.
    hat_x[t] = N == 1 ? x[t] - y[t] : x[t]
  end
  s / 2.
end

################

################ creates a function x'Ax/2 + b'x + c

immutable QuadraticFunction{T<:FloatingPoint, N} <: DifferentiableFunction
  A::Matrix{T}
  b::Vector{T}
  c::T
end

QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}) = QuadraticFunction{T, 1}(A, T[], zero(T))
QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}, b::StridedVector{T}) = QuadraticFunction{T, 2}(A, b, zero(T))
QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}, b::StridedVector{T}, c::T) = QuadraticFunction{T, 3}(A, b, c)

value{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, x::StridedVector{T}) = dot(x, f.A*x) / 2.
value{T<:FloatingPoint}(f::QuadraticFunction{T, 2}, x::StridedVector{T}) = dot(x, f.A*x) / 2. + dot(x, f.b)
value{T<:FloatingPoint}(f::QuadraticFunction{T, 3}, x::StridedVector{T}) = dot(x, f.A*x) / 2. + dot(x, f.b) + f.c

function value{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, x::StridedVector{T}, activeset::ActiveSet)
  A = f.A
  if N > 1
    b = f.b
  end
  indexes = activeset.indexes

  s = zero(T)
  # dot(x, f.A*x) / 2.
  @inbounds for cc=1:activeset.numActive
    ci = indexes[cc]
    @inbounds for rr=1:activeset.numActive
      ri = indexes[rr]
      s += A[ri, ci] * x[ri] * x[ci]
    end
  end
  s /= 2.

  if N > 1
    # dot(x, f.b)
    @inbounds for rr=1:activeset.numActive
      t = indexes[rr]
      s += x[t] * b[t]
    end
  end
  if N > 2
    s += f.c
  end
  s
end

function gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  hat_x
end

function gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  hat_x[:] += f.b
  hat_x
end

function gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::ActiveSet)
  A = f.A
  b = f.b

  indexes = activeset.indexes

  @inbounds for rr=1:activeset.numActive
    ri = indexes[rr]
    hat_x[ri] = zero(T)
    @inbounds for cc=1:activeset.numActive
      ci = indexes[cc]
      hat_x[ri] += A[ri, ci] * x[ci]
    end
    if N > 1
      hat_x[ri] += b[ri]
    end
  end
  hat_x
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, hat_x::StridedVector{T}, x::StridedVector{T})
    A_mul_B!(hat_x, f.A, x)
    dot(hat_x, x) / 2.
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 2}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2.
  hat_x[:] += f.b
  r + dot(x, f.b)
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 3}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2. + f.c
  hat_x[:] += f.b
  r + dot(x, f.b)
end


function value_and_gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::ActiveSet)
  A = f.A
  if N > 1
    b = f.b
  end

  indexes = activeset.indexes

  s = zero(T)
  # dot(x, f.A*x) / 2.
  @inbounds for rr=1:activeset.numActive
    ri = indexes[rr]
    hat_x[ri] = zero(T)
    @inbounds for cc=1:activeset.numActive
      ci = indexes[cc]
      hat_x[ri] += A[ri, ci] * x[ci]
    end
    s += hat_x[ri] * x[ri] / 2.
    if N > 1
      hat_x[ri] += b[ri]
      s += b[ri] * x[ri]
    end
  end
  if N > 2
    s += f.c
  end
  s
end




####################

