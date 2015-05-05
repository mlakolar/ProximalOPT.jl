import HD

facts("proximal_gradient_descent") do

  srand(0)

  n = 500
  p = 2500

  t_beta = sprandn(p,1,0.05)
  t_beta = vec(full(t_beta))
  A = randn(n,p)
  A = A ./ mapslices(norm, A, 1)
  v = sqrt(0.001)*randn(n)
  b = A*t_beta + v

  println("solving instance with $n examples, $p variables")
  println("nnz(x0) = $(countnz(t_beta)); signal-to-noise ratio: $(norm(A*t_beta)^2/norm(v)^2)")

  AA = A'A
  Ab = A'b

  gamma_max = norm(A'*b, Inf)
  gamma = 0.1*gamma_max;

  f = QuadraticFunction(AA, -Ab)
  g = ProxL1(gamma)

#   ################################################# warmup
#   opt = ProximalOptions(;ftol=1e-8,xtol=1e-8,maxiter=300)
#   ## prox grad
#   h_beta = zeros(p)
#   solve!(ProxGradDescent(), h_beta, f, g; options=opt)

#   ## prox grad -- accelerated
#   h_beta1 = zeros(p)
#   solve!(AccProxGradDescent(), h_beta1, f, g; options=opt)

#   ## coordinate descent
#   h_beta2 = spzeros(p,1)
#   lambda = gamma * ones(p)
#   HD.lasso!(h_beta2, AA, Ab, lambda)
#   h_beta2 = vec(full(h_beta2))
#   #################################################################


  opt = ProximalOptions(;ftol=1e-8,xtol=1e-8,maxiter=300)
  ## prox grad
  h_beta = zeros(p)
  tic()
  solve!(ProxGradDescent(), h_beta, f, g; options=opt)
  toc()

  ## prox grad -- accelerated
  h_beta1 = zeros(p)
  tic()
  solve!(AccProxGradDescent(), h_beta1, f, g; options=opt)
  toc()

  ## coordinate descent
  h_beta2 = spzeros(p,1)
  tic()
  lambda = gamma * ones(p)
  HD.lasso!(h_beta2, AA, Ab, lambda)
  h_beta2 = vec(full(h_beta2))
  toc()

  @fact h_beta2 => roughly(h_beta; atol=1e-3)
  @fact h_beta2 => roughly(h_beta1; atol=1e-3)
end
