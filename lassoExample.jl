
import ProximalOPT
import HD
# reload("ProximalOPT")

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

h_beta = zeros(p)
## prox grad
tic()
f = ProximalOPT.quad_f(AA, -Ab)
g = ProximalOPT.prox_l1(gamma)
tr = ProximalOPT.prox_grad!(h_beta, f, g; MAX_ITER=100, show_trace = false, ABSTOL=1e-4, beta=0.9)
prox_res = (copy(h_beta), tr, toc());

h_beta1 = zeros(p)
## prox grad -- accelerated
tic()
f = ProximalOPT.quad_f(AA, -Ab)
g = ProximalOPT.prox_l1(gamma)
tr = ProximalOPT.acc_prox_grad!(h_beta1, f, g; MAX_ITER=100, show_trace = false, ABSTOL=1e-4, beta=0.9)
acc_prox_res = (copy(h_beta1), tr, toc());

h_beta2 = spzeros(p,1)
tic()
lambda = gamma * ones(p)
HD.lasso!(h_beta2, AA, Ab, lambda)
h_beta2 = vec(full(h_beta2));
toc()
