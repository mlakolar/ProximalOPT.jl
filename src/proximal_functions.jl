
##########################################################
#
#  Proximal Operators
#
##########################################################



######  L1 norm

function prox_l1(lambda::Float64)
  g(x::Vector{Float64}) = norm(x, 1)
  @eval prox_g!(hat_x::Vector{Float64}, x::Vector{Float64}, gamma::Float64) = _prox_l1!(hat_x, x, gamma * $lambda)

  ProximalOperator(
    g, prox_g!
    )
end


# computes argmin lambda * |hat_x|_1 + |x-hat_x|^2 / 2
function _prox_l1!(hat_x::Vector{Float64}, x::Vector{Float64}, lambda::Float64)
  for i=1:length(x)
    hat_x[i] = max(0., x[i] - lambda) - max(0., -x[i] - lambda)
  end
  nothing
end


###### L2 norm


function prox_l2(lambda::Float64)
  @eval prox_g!(hat_x::Vector{Float64}, x::Vector{Float64}, gamma::Float64) = _prox_l2!(hat_x, x, gamma * $lambda)
  ProximalOperator(
    norm, prox_g!
    )
end

# computes argmin lambda * |hat_x|_2 + |x-hat_x|^2 / 2
function _prox_l2!(hat_x::Vector{Float64}, x::Vector{Float64}, lambda::Float64)
  tmp = max(1. - lambda / norm(x), 0)
  for i=1:length(x)
    hat_x[i] = tmp * x[i]
  end
  nothing
end
