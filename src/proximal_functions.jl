
##########################################################
#
#  Proximal Operators
#
##########################################################

abstract ProximableFunction

prox!(g::ProximableFunction, hat_x::StridedArray, x::StridedArray) = prox!(g, hat_x, x, 1.0)
prox{T<:FloatingPoint}(g::ProximableFunction, x::StridedArray, γ::T) = prox!(g, similar(x), x, γ)
prox(g::ProximableFunction, x::StridedArray) = prox!(g, similar(x), x, 1.0)


######  zero

immutable ProxZero <: ProximableFunction
end

value{T<:FloatingPoint}(g::ProxZero, x::StridedArray{T}) = zero(T)

function prox!{T<:FloatingPoint}(::ProxZero, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  copy!(out_x, x)
  out_x
end

######  L1 norm  g(x) = λ * ||x||_1

immutable ProxL1{T<:FloatingPoint} <: ProximableFunction
  λ::T
end

ProxL1{T<:FloatingPoint}(λ::T) = ProxL1{T}(λ)

value{T<:FloatingPoint}(g::ProxL1{T}, x::StridedArray{T}) = g.λ * sumabs(x)

function prox!{T<:FloatingPoint}(g::ProxL1{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  for i in eachindex(x)
    @inbounds out_x[i] = max(0., x[i] - c) - max(0., -x[i] - c)
  end
  out_x
end

###### L2 norm   g(x) = λ * ||x||_2

immutable ProxL2{T<:FloatingPoint} <: ProximableFunction
  λ::T
end

ProxL2{T<:FloatingPoint}(λ::T) = ProxL2{T}(λ)

value{T<:FloatingPoint}(g::ProxL2{T}, x::StridedVector{T}) = g.λ * norm(x)

function prox!{T<:FloatingPoint}(g::ProxL2{T}, out_x::StridedVector{T}, x::StridedVector{T}, γ::T)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  tmp = max(1. - c / norm(x), 0.)
  if tmp > 0.
    @inbounds for i in eachindex(x)
      out_x[i] = tmp * x[i]
    end
  else
    fill!(out_x, 0.)
  end
  out_x
end

###### L2 norm squared g(x) = λ * ||x||_2^2

immutable ProxL2Sq{T<:FloatingPoint} <: ProximableFunction
  λ::T
end

ProxL2Sq{T<:FloatingPoint}(λ::T) = ProxL2Sq{T}(λ)

function value{T<:FloatingPoint}(g::ProxL2Sq{T}, x::StridedArray{T})
  g.λ * sumabs2(x)
end

function prox!{T<:FloatingPoint}(g::ProxL2Sq{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  c = 1. / (1. + 2. * c)
  @inbounds for i in eachindex(x)
    out_x[i] = c * x[i]
  end
  out_x
end

####### nuclear norm  g(x) = λ * ||x||_*

immutable ProxNuclear{T<:FloatingPoint} <: ProximableFunction
  λ::T
end

ProxNuclear{T<:FloatingPoint}(λ::T) = ProxNuclear{T}(λ)

value{T<:FloatingPoint}(g::ProxNuclear{T}, x::StridedMatrix{T}) = g.λ * sum(svdvals(x))

function prox!{T<:FloatingPoint}(g::ProxNuclear{T}, out_x::StridedMatrix{T}, x::StridedMatrix{T}, γ::T)
  @assert size(out_x) == size(x)
  U, S, V = svd(x)
  c = g.λ * γ
  @inbounds for i in eachindex(S)
    S[i] = max(zero(T), S[i] - c)
  end
  nr, nc = size(x)
  fill!(out_x, zero(T))
  for k in eachindex(S)
    for c=1:nc
      for r=1:nr
        @inbounds out_x[r, c] += S[k] * U[r, k] * V[c, k]
      end
    end
  end
  out_x
end

###### sum_k g(x_k)

immutable ProxSumProx{P<:ProximableFunction, I} <: ProximableFunction
  intern_prox::P
  groups::Vector{I}
end

ProxL1L2{T<:FloatingPoint, I}(λ::T, groups::Vector{I}) = ProxSumProx{ProxL2{T}, I}(ProxL2{T}(λ), groups)
ProxL1Nuclear{T<:FloatingPoint, I}(λ::T, groups::Vector{I}) = ProxSumProx{ProxNuclear{T}, I}(ProxNuclear{T}(λ), groups)


function value{T<:FloatingPoint}(g::ProxSumProx, x::StridedArray{T})
  intern_prox = g.intern_prox
  groups = g.groups
  v = zero(T)
  for i in eachindex(groups)
    v += value(intern_prox, sub(x, groups[i]))
  end
  v
end

function prox!{T<:FloatingPoint}(g::ProxSumProx, out_x::StridedArray{T}, x::StridedArray{T}, γ::T)
  @assert size(out_x) == size(x)
  intern_prox = g.intern_prox
  groups = g.groups
  for i in eachindex(groups)
    prox!(intern_prox, sub(out_x, groups[i]), sub(x, groups[i]), γ)
  end
  out_x
end

####


