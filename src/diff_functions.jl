
##########################################################
#
#  Create DifferentiableFunction
#
##########################################################

abstract DifferentiableFunction


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
value{T<:FloatingPoint}(f::QuadraticFunction{T, 3}, x::StridedVector{T}) = dot(x, f.A*x) / 2. + dot(x, f.b) + c

function gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  hat_x
end

function gradient!{T<:FloatingPoint, N}(f::QuadraticFunction{T, N}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  @devec hat_x = hat_x + f.b
  hat_x
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 1}, hat_x::StridedVector{T}, x::StridedVector{T})
    A_mul_B!(hat_x, f.A, x)
    dot(hat_x, x) / 2.
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 2}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2.
  @devec hat_x = hat_x + f.b
  r + dot(x, f.b)
end

function value_and_gradient!{T<:FloatingPoint}(f::QuadraticFunction{T, 3}, hat_x::StridedVector{T}, x::StridedVector{T})
  A_mul_B!(hat_x, f.A, x)
  r = dot(hat_x, x) / 2. + f.c
  @devec hat_x = hat_x + f.b
  r + dot(x, f.b)
end


####################

