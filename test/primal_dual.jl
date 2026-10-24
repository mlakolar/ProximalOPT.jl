module PrimalDualTests
using Test, LinearAlgebra, SparseArrays, Random, ProximalBase, ProximalSolvers
const PS = ProximalSolvers
# A function barrier prevents the loop's test closure from being boxed on 1.10.
function step_allocations(s,f,g,hc,D,τ,σ,a,r)
  PS._pd_step!(s,f,g,hc,D,τ,σ,a,r)
  PS._pd_step!(s,f,g,hc,D,τ,σ,a,r)
  @allocated PS._pd_step!(s,f,g,hc,D,τ,σ,a,r)
end

@testset "Condat–Vũ" begin
  rng = MersenneTwister(23)
  opts = ProximalOptions(maxiter=100_000)
  @testset "TV: direct prox and accelerated gradient" begin
    n = 40
    b = randn(rng,n)
    D = spdiagm(n-1,n,0=>-ones(n-1),1=>ones(n-1))
    f, g, h = SqrNormL2(1.,b), ZeroFunction(), NormL1(.7)
    truth = prox(TotalVariation1D(.7),b)
    pg = solve!(zeros(n),f,TotalVariation1D(.7),AccProxGradDescent(),
                ProximalOptions(maxiter=10_000,gradtol=1e-10,ftol=1e-14,xtol=1e-12))
    @test norm(pg.minimizer-truth,Inf) < 1e-7
    for adaptive in (false,true), restart in (false,true)
      x, y = zeros(n), zeros(n-1)
      method = CondatVu(abstol=1e-9,reltol=1e-8,adaptive=adaptive,restart=restart)
      res = solve!(x,f,g,h,D,method,opts;y=y)
      @test res.converged
      @test x ≈ truth atol=2e-6
      @test res.minimizer === x && res.y === y
      @test norm(x-b+D'y) <= res.eps_pri*1.01
      @test maximum(abs,y) <= .7+1e-12
      @test res.r_norm <= res.eps_pri && res.s_norm <= res.eps_dual
      @test res.τ*(.5+res.σ*4) < 1
      @test res.adaptations <= 50
      @test isfinite(res.minimum)
      @test occursin("Condat",sprint(show,res))
      if !restart
        @test res.D_calls == res.Dt_calls == res.iterations+1
        @test res.grad_calls == res.iterations+1
        @test res.prox_calls == 2res.iterations
      end
      # Warm starts retain both iterates and the final steps.
      warm = solve!(x,f,g,h,D,CondatVu(τ=res.τ,σ=res.σ,abstol=1e-9,reltol=1e-8,adaptive=false,restart=false),opts;y=y)
      @test warm.converged
      @test warm.iterations <= 2
    end
    # A valid Fenchel gap for the identity quadratic + TV.
    gap(x,y) = (f(x)+h(D*x) + sum(abs2,D'y)/2 - dot(b,D'y))/max(1.,f(x)+h(D*x))
    res = solve!(zeros(n),f,g,h,D,CondatVu(abstol=1e-8,reltol=1e-7),
                 ProximalOptions(maxiter=100_000,gaptol=1e-8,gapevery=17);gap=gap)
    @test res.converged && 0 <= res.gap <= 1e-8
    @test res.gap ≈ gap(res.minimizer,res.y)
  end
  @testset "exact diagonal solution; sparse primal and Float32" begin
    for T in (Float32,Float64)
      D = Matrix(Diagonal(T[1,2,3,4]))
      b = T[2,-3,.1,0]
      f, g, h = SqrNormL2(one(T),b),NormL1(T(.2)),NormL1(T(.3))
      truth = sign.(b).*max.(abs.(b).-T(.2).-T(.3).*diag(D),0)
      for sparsex in (false,true), sparseD in (false,true)
        A = sparseD ? sparse(D) : D
        x = sparsex ? SparseIterate(T,4) : zeros(T,4)
        res = solve!(x,f,g,h,A,CondatVu(abstol=1e-7,reltol=1e-5),opts)
        @test res.converged
        @test Vector(x) ≈ truth atol=2e-4
        @test eltype(res.minimizer) === T && eltype(res.y) === T
        @test res.τ isa T && res.minimum isa T
      end
    end
  end
  @testset "zero smooth term and rectangular map" begin
    D = reshape([1.,-1,2],3,1)
    # min |x| + ||Dx-b||²/2: soft threshold of D'b by 1, divided by D'D.
    b = [2.,-1,3]
    expected = max(dot(D,b)-1,0)/sum(abs2,D)
    res = solve!(zeros(1),ZeroFunction(),NormL1(1.),SqrNormL2(1.,b),D,
                 CondatVu(abstol=1e-10,reltol=1e-9),opts)
    @test res.converged
    @test only(res.minimizer) ≈ expected atol=1e-7
    for D in (zeros(2,1),spzeros(2,1))
      res = solve!(zeros(1),SqrNormL2(1.,[2.]),NormL1(.5),NormL1(),D,CondatVu(abstol=1e-10,reltol=1e-9),opts)
      @test res.converged
      @test only(res.minimizer) ≈ 1.5 atol=1e-5
    end
  end
  @testset "limits, trace and errors" begin
    f,g,h,D = SqrNormL2(1.,[3.,-4]),ZeroFunction(),NormL1(.5),Matrix(1.0I,2,2)
    res = solve!(zeros(2),f,g,h,D,CondatVu(restart=false),ProximalOptions(prox_calls_limit=1))
    @test res.iterations == 1 && res.prox_calls == 2 && res.call_limit_reached
    res = solve!(zeros(2),f,g,h,D,CondatVu(restart=false),ProximalOptions(time_limit=-1))
    @test res.iterations == 0 && !res.converged
    @test isfinite(res.minimum)
    res = solve!(zeros(2),f,g,h,D,CondatVu(),ProximalOptions(store_trace=true,extended_trace=true,callback=tr->true))
    @test res.iterations == length(res.trace) == 1
    @test haskey(only(res.trace).metadata,"y")
    @test_throws ArgumentError CondatVu(τ=0)
    @test_throws ArgumentError CondatVu(σ=Inf)
    @test_throws ArgumentError CondatVu(L=-1)
    @test_throws ArgumentError CondatVu(abstol=0,reltol=0)
    @test_throws ArgumentError CondatVu(checkevery=0)
    @test_throws ArgumentError solve!(zeros(2),f,g,h,D,CondatVu(τ=2,σ=1))
    cert = solve!(zeros(2),f,g,h,D,CondatVu(),ProximalOptions(gaptol=1e-6,maxiter=10000))
    @test cert.converged && cert.gap <= 1e-6
    # A smooth term needs no implemented conjugate when its gradient provides
    # the Fenchel equality. Zero g may leave the dual infeasible off stationarity.
    cert = duality_gap(LeastSquaresLoss([1.,-2.],D),ZeroFunction(),NormL1(.5),D,[.5,-1.5],[.5,-.5])
    @test cert.gap ≈ 0 atol=1e-12
    @test duality_gap(SqrNormL2(0.),g,h,D,zeros(2),zeros(2)).gap == 0
    @test_throws DimensionMismatch solve!(zeros(3),f,g,h,D,CondatVu())
    @test_throws DimensionMismatch solve!(zeros(2),f,g,h,D,CondatVu();y=zeros(3))
    @test_throws ArgumentError solve!(zeros(2),f,MCP(.1,3.),h,D,CondatVu())
    @test_throws ArgumentError solve!(zeros(2),NormL1(),g,h,D,CondatVu())
    x=zeros(2)
    @test_throws ArgumentError solve!(x,f,g,h,D,CondatVu();y=x)
    @test_throws ArgumentError solve!([NaN,0.],f,g,h,D,CondatVu())
    # Partial explicit steps preserve the bound, including a zero operator.
    for method in (CondatVu(τ=.2),CondatVu(σ=.2))
      @test solve!(zeros(2),f,g,h,D,method,opts).converged
    end
  end
  @testset "allocation-free ordinary dense/sparse steps" begin
    for T in (Float32,Float64), sparseD in (false,true), sparsex in (false,true)
      D = sparseD ? spdiagm(0=>ones(T,10)) : Matrix{T}(I,10,10)
      f,g,h = SqrNormL2(one(T),ones(T,10)),NormL1(T(.1)),NormL1(T(.2))
      s = PS._pd_state(sparsex ? SparseIterate(T,10) : zeros(T,10),zeros(T,10),f,D)
      hc = Conjugate(h)
      @test step_allocations(s,f,g,hc,D,T(.4),T(.4),T(1e-6),T(1e-4)) == 0
    end
  end
end
end
