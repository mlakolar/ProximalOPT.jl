module LineSearchTest

using Test
using Random
using LinearAlgebra
using ProximalBase
using ProximalOPT

Random.seed!(1)

n, p = 100, 50
X = randn(n, p)
Y = X * vcat(randn(5), zeros(p - 5)) + 0.1 * randn(n)
A, b = X'X / n, -X'Y / n
λ = 0.1 * norm(b, Inf)

@testset "solver reuse $(summary(solver))" for solver in (ProximalGradientDescent(), AccProxGradDescent())
  # without a penalty, ProximalGradientDescent reaches the gradient mode of the
  # line search, so state carried over from the first solve would change the second
  f = QuadraticFunction(A, b)
  opt = ProximalOptions(; ftol = 1e-300, xtol = 1e-300, gradtol = 1e-12, maxiter = 5000,
                        store_trace = true, extended_trace = true)

  x1, x2 = zeros(p), zeros(p)
  res1 = solve!(x1, f, ProxZero(), solver, opt)
  res2 = solve!(x2, f, ProxZero(), solver, opt)
  @test x1 == x2
  @test res1.iterations == res2.iterations
  @test res1.backtrack_steps == res2.backtrack_steps
  @test [s.metadata["x"] for s in res1.trace] == [s.metadata["x"] for s in res2.trace]

  @test res1.backtrack_steps >= res1.iterations
  @test occursin("Backtracking steps: $(res1.backtrack_steps)", sprint(show, res1))
end

@testset "TFOCS backtracking switches to gradients near convergence" begin
  f, g = QuadraticFunction(A, b), ProxL1(λ)
  method = ProximalGradientDescent()
  state = ProximalOPT.initial_state(method, ProximalOptions(), f, g, zeros(p))

  simple, rel = Bool[], Float64[]
  grad_f_x = similar(state.x)
  fresh_gradient = true
  for k in 1:100
    ProximalOPT.update_state!(f, g, state, method)
    push!(simple, state.backtrack_simple)
    push!(rel, abs(state.f_x - state.f_x_previous) / max(abs(state.f_x), abs(state.f_x_previous)))
    value_and_gradient!(f, grad_f_x, state.x)
    fresh_gradient &= grad_f_x == state.grad_f_x
  end

  # the function values are compared while they are far apart ...
  @test all(simple[1:10])
  @test all(rel[simple] .>= 1e-10)
  # ... and the gradients for good once they are close
  k = findfirst(!, simple)
  @test k !== nothing
  @test all(.!simple[k:end])
  @test rel[k] < 1e-10
  # the gradient at x is current after every iteration, also when the mode switches
  @test fresh_gradient
end

end
