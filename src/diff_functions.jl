
##########################################################
#
#  Create DifferentiableFunction
#
##########################################################

# creates a function x'Ax/2 + b'x + c
function quad_f(A::ArrayView)
  function f(x::Vector{Float64})
    dot(x, A*x) / 2.
  end
  function g!(hat_x::Vector{Float64}, x::Vector{Float64})
    A_mul_B!(hat_x, A, x)
  end
  function fg!(hat_x::Vector{Float64}, x::Vector{Float64})
    A_mul_B!(hat_x, A, x)
    dot(hat_x, x) / 2.
  end

  DifferentiableFunction(
    f,
    g!,
    fg!
    )
end

function quad_f(A::ArrayView, b::ArrayView, c::Float64=0.)

  function f(x::Vector{Float64})
    dot(x, A*x) / 2. + dot(x, b) + c
  end
  function g!(hat_x::Vector{Float64}, x::Vector{Float64})
    A_mul_B!(hat_x, A, x)
    for i=1:length(x)
      hat_x[i] += b[i]
    end
    nothing
  end
  function fg!(hat_x::Vector{Float64}, x::Vector{Float64})
    A_mul_B!(hat_x, A, x)
    r = dot(hat_x, x) / 2. + c
    for i=1:length(x)
      hat_x[i] += b[i]
    end
    r + dot(x, b)
  end

  DifferentiableFunction(
    f,
    g!,
    fg!
    )
end
