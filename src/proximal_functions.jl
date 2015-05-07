
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
    @inbounds out_x[i] = shrink(x[i], c)
  end
  out_x
end

function value{T<:FloatingPoint}(g::ProxL1{T}, x::StridedArray{T}, activeset::ActiveSet)
  r = zero(T)
  allCoordinate = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    t = allCoordinate[i]
    r += abs(x[t])
  end
  g.λ * r
end

function prox!{T<:FloatingPoint}(g::ProxL1{T}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T, activeset::ActiveSet)
  @assert size(out_x) == size(x)
  c = g.λ * γ
  allCoordinate = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    t = allCoordinate[i]
    out_x[t] = shrink(x[t], c)
  end
  out_x
end

function active_set{T<:FloatingPoint}(::ProxL1{T}, x::StridedArray{T}; zero_thr::T=1e-4)
  numElem = length(x)
  activeset = [1:numElem;]
  numActive = 0
  for j = 1:numElem
    if abs(x[j]) >= zero_thr
      numActive += 1
      activeset[numActive], activeset[j] = activeset[j], activeset[numActive]
    end
  end
  ActiveSet(activeset, numActive)
end


# active set of coordinates
# prox function
# differentiable function
# current iterate
# tmp array
function add_violator!{T<:FloatingPoint}(
    activeset::ActiveSet, x::StridedArray{T},
    g::ProxL1{T}, f::DifferentiableFunction, tmp::StridedArray{T}; zero_thr::T=1e-4, grad_tol=1e-6
    )
  λ = g.λ
  changed = false

  numElem = length(x)
  numActive = activeset.numActive
  allCoordinate = activeset.allCoordinate
  # check for things to be removed from the active set
  i = 0
  while i < numActive
    i = i + 1
    t = allCoordinate[i]
    if abs(x[t]) < zero_thr
      x[t] = zero(T)
      changed = true
      allCoordinate[numActive], allCoordinate[i] = allCoordinate[i], allCoordinate[numActive]
      numActive -= 1
      i = i - 1
    end
  end

  gradient!(f, tmp, x)
  I = 0
  V = zero(T)
  for i=numActive+1:numElem
    t = allCoordinate[i]
    nV = abs(tmp[t])
    if V < nV
      I = i
      V = nV
    end
  end
  if I > 0 && V + grad_tol > λ
    changed = true
    numActive += 1
    allCoordinate[numActive], allCoordinate[I] = allCoordinate[I], allCoordinate[numActive]
  end
  activeset.numActive = numActive
  changed
end

function interpolation_point!{T<:FloatingPoint}(
    g::ProxL1{T}, tmp::StridedArray{T}, y::StridedArray{T}, grad_y::StridedArray{T}, λ::T, activeset::ActiveSet
    )
  allCoordinate = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    t = allCoordinate[i]
    tmp[t] = y[t] - λ * grad_y[t]
  end
  nothing
end

function taylor_value{T<:FloatingPoint}(
    g::ProxL1{T}, fy::T, z::StridedArray{T}, y::StridedArray{T}, grad_y::StridedArray{T}, λ::T, activeset::ActiveSet
    )
  allCoordinate = activeset.allCoordinate
  fval = fy
  dgh = zero(T)
  nhsq = zero(T)
  @inbounds for i=1:activeset.numActive
    t = allCoordinate[i]
    h = z[t] - y[t]
    dgh += grad_y[t] * h
    nhsq += h^2.
  end
  fval + dgh + nhsq / 2. / λ
end

function update_y{T<:FloatingPoint}(
    g::ProxL1{T}, y::StridedArray{T}, z::StridedArray{T}, x::StridedArray{T}, ω::T, activeset::ActiveSet
    )
  allCoordinate = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    t = allCoordinate[i]
    y[t] = (1.+ω)*z[t] - ω*x[t]
  end
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


#

function active_set{T<:FloatingPoint, I}(g::ProxSumProx{ProxNuclear{T}, I}, x::StridedArray{T}; zero_thr::T=1e-4)
  groups = g.groups
  numElem = length(groups)
  activeset = [1:numElem;]
  numActive = 0
  for j = 1:numElem
    if vecnorm(sub(x, groups[j])) > zero_thr
      numActive += 1
      activeset[numActive], activeset[j] = activeset[j], activeset[numActive]
    end
  end
  ActiveSet(activeset, numActive)
end


function value{T<:FloatingPoint, I}(g::ProxSumProx{ProxNuclear{T}, I}, x::StridedArray{T}, activeset::ActiveSet)
  v = zero(T)
  intern_prox = g.intern_prox
  groups = g.groups
  activeGroups = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    indGroup = activeGroups[i]
    v += value(intern_prox, sub(x, groups[indGroup]))
  end
  v
end


function prox!{T<:FloatingPoint, I}(g::ProxSumProx{ProxNuclear{T}, I}, out_x::StridedArray{T}, x::StridedArray{T}, γ::T, activeset::ActiveSet)
  @assert size(out_x) == size(x)
  intern_prox = g.intern_prox
  groups = g.groups
  activeGroups = activeset.allCoordinate
  @inbounds for i=1:activeset.numActive
    indGroup = activeGroups[i]
    prox!(intern_prox, sub(out_x, groups[indGroup]), sub(x, groups[indGroup]), γ)
  end
  out_x
end




function add_violator!{T<:FloatingPoint, I}(
    activeset::ActiveSet, x::StridedArray{T},
    g::ProxSumProx{ProxNuclear{T}, I}, f::DifferentiableFunction, tmp::StridedArray{T}; zero_thr::T=1e-4, grad_tol=1e-6
    )
  λ = g.intern_prox.λ
  groups = g.groups
  numElem = length(groups)
  changed = false

  numActive = activeset.numActive
  activeGroups = activeset.allCoordinate
  # check for things to be removed from the active set
  i = 0
  while i < numActive
    i = i + 1
    t = activeGroups[i]
    xt = sub(x, groups[t])
    if vecnorm(xt) < zero_thr
      fill!(xt, zero(T))
      changed = true
      activeGroups[numActive], activeGroups[i] = activeGroups[i], activeGroups[numActive]
      numActive -= 1
      i = i - 1
    end
  end

  gradient!(f, tmp, x)
  I = 0
  V = zero(T)
  for i=numActive+1:numElem
    t = activeGroups[i]
    gxt = sub(tmp, groups[t])
    nV = sqrt(eigmax(gxt'*gxt))
    if V < nV
      I = i
      V = nV
    end
  end
  if I > 0 && V + grad_tol > λ
    changed = true
    numActive += 1
    activeGroups[numActive], activeGroups[I] = activeGroups[I], activeGroups[numActive]
  end
  activeset.numActive = numActive
  changed
end


function interpolation_point!{T<:FloatingPoint, I}(
    g::ProxSumProx{ProxNuclear{T}, I}, tmp::StridedArray{T}, y::StridedArray{T}, grad_y::StridedArray{T}, λ::T, activeset::ActiveSet
    )
  groups = g.groups
  activeGroups = activeset.allCoordinate
  @inbounds for ai=1:activeset.numActive
    t = activeGroups[ai]
    tmpt = sub(tmp, groups[t])
    yt = sub(y, groups[t])
    grad_yt = sub(grad_y, groups[t])
    @inbounds for i in eachindex(tmpt)
      tmpt[i] = yt[i] - λ * grad_yt[i]
    end
  end
  nothing
end

function taylor_value{T<:FloatingPoint, I}(
    g::ProxSumProx{ProxNuclear{T}, I}, fy::T, z::StridedArray{T}, y::StridedArray{T}, grad_y::StridedArray{T}, λ::T, activeset::ActiveSet
    )
  groups = g.groups
  activeGroups = activeset.allCoordinate
  dgh = zero(T)
  nhsq = zero(T)
  @inbounds for ai=1:activeset.numActive
    t = activeGroups[ai]
    zt = sub(z, groups[t])
    yt = sub(y, groups[t])
    grad_yt = sub(grad_y, groups[t])
    @inbounds for i in eachindex(zt)
      h = zt[i] - yt[i]
      dgh += grad_yt[i] * h
      nhsq += h^2.
    end
  end
  fy + dgh + nhsq / 2. / λ
end

function update_y{T<:FloatingPoint, I}(
    g::ProxSumProx{ProxNuclear{T}, I}, y::StridedArray{T}, z::StridedArray{T}, x::StridedArray{T}, ω::T, activeset::ActiveSet
    )
  groups = g.groups
  activeGroups = activeset.allCoordinate
  @inbounds for ai=1:activeset.numActive
    t = activeGroups[ai]
    xt = sub(x, groups[t])
    yt = sub(y, groups[t])
    zt = sub(z, groups[t])
    @inbounds for i in eachindex(zt)
      yt[i] = (1.+ω)*zt[i] - ω*xt[i]
    end
  end
end


####


