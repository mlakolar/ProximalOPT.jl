
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

function prox_l1l2(groups::Vector{UnitRange{Int64}}, lambda::Vector{Float64})
  function g(x::ArrayView)
    r = 0.
    for i=1:length(lambda)
      tr = 0.
      for j=groups[i]
        tr += x[j]^2
      end
      r += lambda[i] * sqrt(tr)
    end
    r
  end
  prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_l1l2!(hat_x, x, groups, lambda, gamma)
  ProximalOperator(
    g, prox_g!
    )
end

function _prox_l1l2!(hat_x::ArrayView, x::ArrayView, groups::Vector{UnitRange{Int64}}, lambda::Vector{Float64}, gamma::Float64)
  for i=1:length(lambda)
    hat_x_i = view(hat_x, groups[i])
    x_i = view(x, groups[i])
    _prox_l2!(hat_x_i, x_i, lambda[i]*gamma)
  end
  nothing
end

###### L2 norm -- squared

function prox_l2sq(lambda::Float64)
  g(x::ArrayView) = lambda * dot(x, x)
  @eval prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_l2sq!(hat_x, x, gamma * $lambda)
  ProximalOperator(
    g, prox_g!
    )
end

# computes argmin lambda * |hat_x|_2^2 + |x-hat_x|^2 / 2
function _prox_l2sq!(hat_x::ArrayView, x::ArrayView, lambda::Float64)
  tmp = 1. / (1. + 2. * lambda)
  for i=1:length(x)
    hat_x[i] = tmp * x[i]
  end
  nothing
end


####### nuclear norm

function prox_nuclear(lambda::Float64)
  g(x::ArrayView) = lambda * sum(svdvals(x))
  @eval prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_nuclear!(hat_x, x, gamma * $lambda)
  ProximalOperator(
    g, prox_g!
    )
end

function _prox_nuclear!(hat_x::ArrayView, x::ArrayView, lambda::Float64)
  U, S, V = svd(x)
  for i=1:length(S)
    S[i] = max(0., S[i] - lambda) - max(0., -S[i] - lambda)
  end
  tmp_x = U * (Diagonal(S) * V')
  nr, nc = size(x)
  for r=1:nr
    for c=1:nc
      hat_x[r,c] = tmp_x[r, c]
    end
  end
  nothing
end


function prox_l1nuclear(groups::Vector{UnitRange{Int64}}, lambda::Vector{Float64})
  function g(x::ArrayView)
    r = 0.
    for i=1:length(lambda)
      tmp_v = view(x, :, groups[i])
      r += lambda[i] * sum(svdvals(tmp_v))
    end
    r
  end
  prox_g!(hat_x::ArrayView, x::ArrayView, gamma::Float64) = _prox_l1nuclear!(hat_x, x, groups, lambda, gamma)
  ProximalOperator(
    g, prox_g!
    )
end

function _prox_l1nuclear!(hat_x::ArrayView, x::ArrayView, groups::Vector{UnitRange{Int64}}, lambda::Vector{Float64}, gamma::Float64)
  for i=1:length(lambda)
    hat_x_i = view(hat_x, :, groups[i])
    x_i = view(x, :, groups[i])
    _prox_nuclear!(hat_x_i, x_i, lambda[i]*gamma)
  end
  nothing
end

