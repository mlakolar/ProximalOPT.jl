
##########################################################
#
#  Proximal Operators
#
##########################################################



######  L1 norm

function prox_l1(lambda::Float64)
  g(x::ArrayView) = lambda * norm(x, 1)
  @eval prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_l1!(hat_x, x, gamma * $lambda)

  ProximalOperator(
    g, prox_g!
    )
end


# computes argmin lambda * |hat_x|_1 + |x-hat_x|^2 / 2
function _prox_l1!(hat_x::ArrayView, x::ArrayView, lambda::Float64)
  for i=1:length(x)
    hat_x[i] = max(0., x[i] - lambda) - max(0., -x[i] - lambda)
  end
  nothing
end


###### L2 norm

function prox_l2(lambda::Float64)
  g(x::ArrayView) = lambda * norm(x)
  @eval prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_l2!(hat_x, x, gamma * $lambda)
  ProximalOperator(
    g, prox_g!
    )
end

# computes argmin lambda * |hat_x|_2 + |x-hat_x|^2 / 2
function _prox_l2!(hat_x::ArrayView, x::ArrayView, lambda::Float64)
  tmp = max(1. - lambda / norm(x), 0.)
  if tmp > 0.
    for i=1:length(x)
      hat_x[i] = tmp * x[i]
    end
  else
    fill!(hat_x, 0.)
  end
  nothing
end

function prox_l1l2(groups::Array{Array{Int64, 1}, 1}, lambda::Vector{Float64})
  function g(x::ArrayView)
    r = 0.
    for i=1:length(lambda)
      r += lambda[i] * norm(x[:])
    end
  end
  prox_g!(hat_x::Vector{ArrayView}, x::Vector{ArrayView}, gamma::Float64) = _prox_l1l2!(hat_x, x, lambda, gamma)
  ProximalOperator(
    g, prox_g!
    )
end

function _prox_l1l2!(hat_x::Vector{ArrayView}, x::Vector{ArrayView}, lambda::Vector{Float64}, gamma::Float64)
  for i=1:length(hat_x)
    _prox_l2!(hat_x[i], x[i], lambda[i]*gamma)
  end
  nothing
end
