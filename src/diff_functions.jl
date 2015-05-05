
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

value{T<:FloatingPoint}(f::L2Loss{T, 0}, x::StridedArray{T}) = sumabs2(x) / 2.

function value{T<:FloatingPoint}(f::L2Loss{T, 1}, x::StridedArray{T})
  y = f.y
  @assert size(y) == size(x)
  s = zero(T)
  @inbounds for i in eachindex(x)
    s += (x[i]-y[i])^2.
  end
  s / 2.
end

function value{T<:FloatingPoint}(f::L2Loss{T, 0}, x::StridedArray{T}, activeset::Vector{Int64})
  s = zero(T)
  @inbounds for ind in eachindex(activeset)
    s += x[activeset[ind]]^2.
  end
  s / 2.
end

function value{T<:FloatingPoint}(f::L2Loss{T, 1}, x::StridedArray{T}, activeset::Vector{Int64})
  y = f.y
  @assert size(y) == size(x)
  s = zero(T)
  @inbounds for ind in eachindex(activeset)
    s += (x[activeset[ind]]-y[activeset[ind]])^2.
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

function value_and_gradient!{T<:FloatingPoint}(f::L2Loss{T, 0}, hat_x::StridedArray{T}, x::StridedArray{T})
  copy!(hat_x, x)
  sumabs2(x) / 2.
end
function value_and_gradient!{T<:FloatingPoint}(f::L2Loss{T, 1}, hat_x::StridedArray{T}, x::StridedArray{T})
  y = f.y
  @assert size(y) == size(x)
  for i in eachindex(x)
    hat_x[i] = x[i]-y[i]
  end
  sumabs2(hat_x) / 2.
end

################

################ creates a function x'Ax/2 + b'x + c

immutable QuadraticFunction{T<:FloatingPoint, N} <: DifferentiableFunction
  A::StridedMatrix{T}
  b::StridedVector{T}
  c::T
end

QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}) = QuadraticFunction{T, 1}(A, T[], zero(T))
QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}, b::StridedVector{T}) = QuadraticFunction{T, 2}(A, b, zero(T))
QuadraticFunction{T<:FloatingPoint}(A::StridedMatrix{T}, b::StridedVector{T}, c::T) = QuadraticFunction{T, 3}(A, b, c)

value{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, x::StridedVector{T}) = dot(x, f.A*x) / 2.
value{T<:FloatingPoint}(f::QuadraticFunction{T, 2}, x::StridedVector{T}) = dot(x, f.A*x) / 2. + dot(x, f.b)
value{T<:FloatingPoint}(f::QuadraticFunction{T, 3}, x::StridedVector{T}) = dot(x, f.A*x) / 2. + dot(x, f.b) + f.c

function value{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, x::StridedVector{T}, activeset::Vector{Int64})
  A = f.A
  b = f.b

  s = zero(T)
  # dot(x, f.A*x) / 2.
  @inbounds for ci in eachindex(activeset)
    @inbounds for ri in eachindex(activeset)
      s += A[activeset[ri], activeset[ci]] * x[activeset[ri]] * x[activeset[ci]]
    end
  end
  s /= 2.

  if N > 1
    # dot(x, f.b)
    @inbounds for i in eachindex(activeset)
      s += x[activeset[i]] * b[activeset[i]]
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

function gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::Vector{Int64})
  A = f.A
  b = f.b

  @inbounds for ri in eachindex(activeset)
    hat_x[activeset[ri]] = zero(T)
    @inbounds for ci in eachindex(activeset)
      hat_x[activeset[ri]] += A[activeset[ri], activeset[ci]] * x[activeset[ci]]
    end
    if N > 1
      hat_x[activeset[ri]] += b[activeset[ri]]
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

function value_and_gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T}, activeset::Vector{Int64})
  A = f.A
  b = f.b

  s = zero(T)
  # dot(x, f.A*x) / 2.
  @inbounds for ri in eachindex(activeset)
    hat_x[activeset[ri]] = zero(T)
    @inbounds for ci in eachindex(activeset)
      hat_x[activeset[ri]] += A[activeset[ri], activeset[ci]] * x[activeset[ci]]
    end
    s += hat_x[activeset[ri]] * x[activeset[ri]] / 2.
    if N > 1
      hat_x[activeset[ri]] += b[activeset[ri]]
      s += b[activeset[ri]] * x[activeset[ri]]
    end
  end
  if N > 2
    s += f.c
  end
  s
end


####################

