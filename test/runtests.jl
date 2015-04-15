using FactCheck

import ProximalOPT
using ArrayViews


facts("proximal_l2") do

  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    xv = view(x, :)
    hat_xv = view(hat_x, :)

    lambda = norm(x) + 0.1
    # check the norm
    prox_op = ProximalOPT.prox_l2(lambda)
    @fact prox_op.g(xv) => roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact prox_op.g(xv) => roughly(lambda * sqrt(ln))

    # check shrinkage
    prox_op.prox_g!(hat_xv, xv, 1.)

    @fact hat_x => roughly(zeros(Float64, p))
    @fact norm(hat_xv) => roughly(0.)
  end

  context("shrink not to zero") do
    p = 2
    x = [1., 2.]
    hat_x = randn(p)

    xv = view(x, :)
    hat_xv = view(hat_x, :)

    lambda = 1.
    # check the norm
    prox_op = ProximalOPT.prox_l2(lambda)

    @fact prox_op.g(xv) => roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact prox_op.g(xv) => roughly(lambda * sqrt(ln))

    # check shrinkage
    prox_op.prox_g!(hat_xv, xv, 1.)

    @fact hat_x => roughly((1.-1./sqrt(5.))*xv)
    @fact norm(hat_x) => roughly(sqrt(5.)-1.)
  end

end

facts("proximal_l1l2") do

  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    groups = Array(UnitRange{Int64}, 2)
    groups[1] = 1:5
    groups[2] = 6:10

    xv = view(x, :)
    hat_xv = view(hat_x, :)

    lambda = norm(x) + 0.1
    lambda = [lambda, lambda+0.1]

    # check the norm
    prox_op = ProximalOPT.prox_l1l2(groups, lambda)
    @fact prox_op.g(xv) => roughly(lambda[1] * norm(x[1:5]) + lambda[2] * norm(x[6:10]))

    # check shrinkage
    prox_op.prox_g!(hat_xv, xv, 1.)

    @fact hat_x => roughly(zeros(Float64, p))
    @fact norm(hat_xv) => roughly(0.)
  end

  context("shrink not to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    groups = Array(UnitRange{Int64}, 2)
    groups[1] = 1:5
    groups[2] = 6:10

    xv = view(x, :)
    hat_xv = view(hat_x, :)

    lambda = [norm(x[1:5]) - 0.2, norm(x[6:10]) - 0.2]

    # check the norm
    prox_op = ProximalOPT.prox_l1l2(groups, lambda)
    @fact prox_op.g(xv) => roughly(lambda[1] * norm(x[1:5]) + lambda[2] * norm(x[6:10]))

    # check shrinkage
    prox_op.prox_g!(hat_xv, xv, 1.)

    @fact hat_x[1:5] => roughly( (1.-lambda[1]/norm(x[1:5])) * x[1:5] )
    @fact hat_x[6:10] => roughly( (1.-lambda[2]/norm(x[6:10])) * x[6:10] )
  end

end

facts("proximal_l1") do

end


facts("proximal_l1sq") do
  n = 100
  p = 10

  X = randn(n, p)
  Y = X * ones(p) + 0.1*randn(n)

  XX = X'X / n
  Xyn = - X'Y / n

  # solve ols
  hat_beta = zeros(p)
  f = ProximalOPT.quad_f(view(XX, :, :), view(Xyn, :))
  prox_op = ProximalOPT.prox_l2sq(0.)
  ProximalOPT.prox_grad!(hat_beta, f, prox_op; ABSTOL=1e-10)
  @fact hat_beta => roughly(-XX \ Xyn; atol=1e-5)

  # solve ridge regression
  lambda = 1.
  hat_beta = zeros(p)
  f = ProximalOPT.quad_f(view(XX, :, :), view(Xyn, :))
  prox_op = ProximalOPT.prox_l2sq(lambda)
  ProximalOPT.prox_grad!(hat_beta, f, prox_op; ABSTOL=1e-10)

  @fact hat_beta => roughly(-(XX + 2*lambda*eye(p)) \ Xyn; atol=1e-5)
end



FactCheck.exitstatus()


