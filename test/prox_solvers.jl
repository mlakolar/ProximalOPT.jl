facts("active proximal_gradient_descent") do

  y = [1., 2., 3.]
  x = zeros(3)

  f = L2Loss(y)
  g = ProxL1(1.5)
  solve!(ActiveAccProxGradDescent(), x, f, g)

  @fact x => roughly([0, 0.5, 1.5])
end


facts("proximal_gradient_descent") do

  context("L2Loss + nuclear norm") do
    # one group
    x = randn(10, 10)
    y = randn(30, 10)
    y = y'y

    f = L2Loss(y)
    g = ProxNuclear(0.1)

    U, S, V = svd(y)
    for k=1:10
      S[k] = max(S[k] - 0.1, 0.)
    end
    tx = U * (Diagonal(S)*V')

    solve!(ProxGradDescent(), x, f, g)
    @fact x => roughly(tx; atol=1e-3)

    # two groups

    x = randn(10, 20)

    t = randn(30, 10)
    y = t't / 30
    t = randn(30, 10)
    y = hcat(y, t't / 30)

    lambda = 0.2
    f = L2Loss(y)
    groups = [(1:10, 1:10), (1:10,11:20)]
    g = ProxL1Nuclear(lambda, groups)

    tx = similar(x)
    # construct true x
    for i = 1:length(groups)
      U, S, V = svd(sub(y, groups[i]))
      for k=1:length(S)
        S[k] = max(S[k] - lambda, 0.)
      end
      txv = sub(tx, groups[i])
      txv[:] = U * Diagonal(S) * V'
    end

    solve!(ProxGradDescent(), x, f, g)
    @fact x => roughly(tx; atol=1e-3)
  end

  context("L2Loss + no penalty") do
    x = randn(5)
    y = randn(5)

    f1 = L2Loss()
    f2 = L2Loss(y)
    g = ProxZero()

    solve!(ProxGradDescent(), x, f1, g)
    @fact norm(x) => roughly(0.; atol=1e-5)

    solve!(ProxGradDescent(), x, f2, g)
    @fact x => roughly(y; atol=1e-5)
  end

  context("L2Loss + l1 penalty") do
    opt = ProximalOptions(;ftol=1e-12,xtol=1e-12,maxiter=1000)
    x = randn(5)
    y = randn(5) + 1.

    f = L2Loss(y)
    g = ProxL1(0.1)

    solve!(ProxGradDescent(), x, f, g; options=opt)
    x2 = similar(x)
    prox!(g, x2, y)
    @fact x => roughly(x2;atol=1e-5)
  end

  context("no penalty") do

    A = randn(20, 10)
    A = A'A
    b = randn(10)
    c = 1.

    f = QuadraticFunction(A, b, c)
    g = ProxZero()

    x = randn(10)
    solve!(ProxGradDescent(), x, f, g)
    tx = -A \ b
    @fact abs(value(f, tx) - value(f, x)) / value(f, tx) < 1e-3 => true

    x = randn(10)
    solve!(AccProxGradDescent(), x, f, g)
    tx = -A \ b
    @fact abs(value(f, tx) - value(f, x)) / value(f, tx) < 1e-3 => true
  end


  context("ridge regression") do
    n = 200
    p = 30

    X = randn(n, p)
    Y = X * ones(p) + 0.1*randn(n)

    true_beta = X \ Y

    f = QuadraticFunction(X'X / n, -X'Y / n)

    # solve ols
    g = ProxL2Sq(0.)
    hat_beta = zeros(p)
    solve!(ProxGradDescent(), hat_beta, f, g)
    @fact abs(value(f, true_beta) - value(f, hat_beta)) / value(f, true_beta) < 1e-3 => true

    hat_beta = zeros(p)
    solve!(AccProxGradDescent(), hat_beta, f, g)
    @fact abs(value(f, true_beta) - value(f, hat_beta)) / value(f, true_beta) < 1e-3 => true


    # solve ridge regression
    ridge_beta = (X'X / n + 2 * eye(p)) \ (X'Y / n)
    g = ProxL2Sq(1.)
    hat_beta = zeros(p)
    solve!(ProxGradDescent(), hat_beta, f, g)
    @fact abs(value(f, ridge_beta) - value(f, hat_beta)
              + value(g, ridge_beta) - value(g, hat_beta)) / (value(f, ridge_beta) + value(g, hat_beta)) < 1e-3 => true


    hat_beta = zeros(p)
    solve!(AccProxGradDescent(), hat_beta, f, g)
    @fact abs(value(f, ridge_beta) - value(f, hat_beta)
              + value(g, ridge_beta) - value(g, hat_beta)) / (value(f, ridge_beta) + value(g, hat_beta)) < 1e-3 => true
    @fact hat_beta => roughly(ridge_beta; atol=1e-3)

  end

end
