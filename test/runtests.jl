using FactCheck

using ProximalOPT
using ArrayViews


facts("proximal_l2") do

  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    lambda = norm(x) + 0.1
    g = ProxL2(lambda)
    # check the norm
    @fact value(g, x) => roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact value(g, x) => roughly(lambda * sqrt(ln))

    # check shrinkage
    prox!(g, hat_x, x, 1.)
    @fact hat_x => roughly(zeros(Float64, p))
    @fact norm(hat_x) => roughly(0.)
  end

  context("shrink not to zero") do
    p = 2
    x = [1., 2.]
    hat_x = randn(p)

    lambda = 1.
    # check the norm
    g = ProxL2(lambda)

    @fact value(g, x) => roughly(lambda * norm(x))
    ln = 0.
    for j=1:p
      ln += x[j]^2
    end
    @fact value(g, x) => roughly(lambda * sqrt(ln))

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @fact hat_x => roughly((1.-1./sqrt(5.))*x)
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

    lambda = norm(x) + 0.1

    # check the norm
    g = ProxL1L2(lambda, groups)
    @fact value(g, x) => roughly(lambda * norm(x[1:5]) + lambda * norm(x[6:10]))

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @fact hat_x => roughly(zeros(Float64, p))
    @fact norm(hat_x) => roughly(0.)
  end

  context("shrink not to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)

    groups = Array(UnitRange{Int64}, 2)
    groups[1] = 1:5
    groups[2] = 6:10

    lambda = min(norm(x[1:5]), norm(x[6:10])) - 0.2

    # check the norm
    g = ProxL1L2(lambda, groups)
    @fact value(g, x) => roughly(lambda * norm(x[1:5]) + lambda * norm(x[6:10]))

    # check shrinkage
    prox!(g, hat_x, x, 1.)

    @fact hat_x[1:5] => roughly( (1.-lambda/norm(x[1:5])) * x[1:5] )
    @fact hat_x[6:10] => roughly( (1.-lambda/norm(x[6:10])) * x[6:10] )
  end

end

# facts("proximal_l1") do

# end


facts("proximal_l1sq") do

  context("prox operator") do
    x = [1., 2.]
    lambda = 1.

    # norm value
    g = ProxL2Sq(0.)
    @fact value(g, x) => 0.

    g = ProxL2Sq(1.)
    @fact value(g, x) => sumabs2(x)

    g = ProxL2Sq(2.)
    @fact value(g, x) => roughly(2. * norm(x)^2)

    # prox
    hat_x = similar(x)

    g = ProxL2Sq(0.)
    @fact prox!(g, hat_x, x) => roughly(x)

    g = ProxL2Sq(1.)
    @fact prox!(g, hat_x, x) => roughly(x / 3.)

    g = ProxL2Sq(2.)
    @fact prox!(g, hat_x, x) => roughly(x / 5.)


  end

  context("ridge regression") do
    n = 100
    p = 10

    X = randn(n, p)
    Y = X * ones(p) + 0.1*randn(n)

    XX = X'X / n
    Xyn = - X'Y / n

    # solve ols
    hat_beta = zeros(p)
    f = QuadraticFunction(XX, Xyn)
    g = ProxL2Sq(0.)
    ProximalOPT.prox_grad!(hat_beta, f, g; ABSTOL=1e-10)
    @fact hat_beta => roughly(-XX \ Xyn; atol=1e-5)

    # solve ridge regression
    lambda = 1.
    hat_beta = zeros(p)
    f = QuadraticFunction(XX, Xyn)
    g = ProxL2Sq(lambda)
    ProximalOPT.prox_grad!(hat_beta, f, g; ABSTOL=1e-10)

    @fact hat_beta => roughly(-(XX + 2*lambda*eye(p)) \ Xyn; atol=1e-5)

  end
end



FactCheck.exitstatus()


