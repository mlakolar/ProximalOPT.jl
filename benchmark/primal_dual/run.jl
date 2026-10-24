# Run in this directory's independent environment. No oracle/benchmark packages
# become dependencies of ProximalSolvers. CSV output includes failures/limits.
using LinearAlgebra, SparseArrays, Random, Printf, Test
using ProximalBase, ProximalSolvers, ProximalAlgorithms, ProximalCore
import JuMP, HiGHS
include(joinpath(@__DIR__,"..","proxalg","adapter.jl"))
const PS = ProximalSolvers
const PA = ProximalAlgorithms
BLAS.set_num_threads(1)
const OUT = get(ENV,"PD_OUT",joinpath(@__DIR__,"results"))
mkpath(OUT)
const ORACLE_ONLY = get(ENV,"PD_ORACLE_ONLY","false") == "true"
const MAXTIME = parse(Float64,get(ENV,"PD_MAXTIME","15"))

function quantile_lp(X,b,q,λ)
  n,p = size(X)
  m=JuMP.Model(HiGHS.Optimizer); JuMP.set_silent(m)
  JuMP.set_attribute(m,"primal_feasibility_tolerance",1e-9)
  JuMP.set_attribute(m,"dual_feasibility_tolerance",1e-9)
  JuMP.@variable(m,bp[1:p]>=0); JuMP.@variable(m,bm[1:p]>=0)
  JuMP.@variable(m,u[1:n]>=0); JuMP.@variable(m,v[1:n]>=0)
  eq=JuMP.@constraint(m,X*(bp-bm)+u-v .== b)
  JuMP.@objective(m,Min,(q*sum(u)+(1-q)*sum(v))/n+λ*(sum(bp)+sum(bm)))
  JuMP.optimize!(m)
  @test JuMP.is_solved_and_feasible(m;dual=true)
  # Our y is the subgradient with respect to the predictions, the negative
  # of the LP's residual/rank-score dual.
  JuMP.value.(bp)-JuMP.value.(bm), -JuMP.dual.(eq), JuMP.objective_value(m)
end

# Identical gap checks and caps for both packages. The adapter keeps its
# upstream unbuffered conjugate so allocation comparisons use the existing API.
function pa_gap(alg,f,g,h,D,steps,tol;kwargs...)
  k=Ref(0); t0=time()
  stop(iter,state)=begin
    k[]+=1
    if k[]%25 == 0
      x,y=PA.default_solution(iter,state)
      cert=duality_gap(f,g,h,D,x,y)
      cert.gap/max(abs(cert.primal),floatmin(Float64)) <= tol && return true
    end
    time()-t0 > MAXTIME
  end
  alg(maxit=1_000_000,stop=stop)(x0=zeros(size(D,2)),y0=zeros(size(D,1)),
                               g=PAProx(g),h=PAProx(h),L=D,gamma=steps;kwargs...)
end

function quantile_data(n,p,q=.3)
  rng=MersenneTwister(23+n+p)
  X=randn(rng,n,p); X ./= sqrt.(sum(abs2,X;dims=1)./n)
  β=zeros(p); β[1:min(5,p)] .= 1
  b=X*β+randn(rng,n)
  λ=.1*norm(X' * ifelse.(b .> 0,q,q-1),Inf)/n
  X,b,λ
end

@testset "Quantile LP primal and dual references" begin
  for q in (.25,.5,.75), sparseD in (false,true)
    X,b,λ=quantile_data(60,12,q)
    D=sparseD ? sparse(X) : X
    xlp,ylp,Flp=quantile_lp(X,b,q,λ)
    f,g,h=ZeroFunction(),NormL1(λ),Scaled(Translate(CheckLoss(1-q),b),1/length(b))
    res=solve!(zeros(12),f,g,h,D,CondatVu(abstol=1e-10,reltol=1e-9,normD=opnorm(X)),
               ProximalOptions(maxiter=500_000,gaptol=1e-9,gapevery=25,time_limit=60))
    @test res.converged
    @test abs(res.minimum-Flp) <= 2e-8
    @test norm(res.minimizer-xlp,Inf) < 1e-5
    @test norm(res.y-ylp,Inf) < 2e-6
    @test norm(X'res.y,Inf) <= λ+1e-7
    @test all(-q/60-1e-9 .<= res.y .<= (1-q)/60+1e-9)
    @printf("ORACLE q=%.2f sparse=%s iterations=%d ΔF=%.3e Δx=%.3e Δdual=%.3e gap=%.3e\n",q,sparseD,res.iterations,res.minimum-Flp,norm(res.minimizer-xlp,Inf),norm(res.y-ylp,Inf),res.gap)
  end
end
@testset "Generalized lasso: HiGHS quadratic program" begin
  rng=MersenneTwister(2323); n,p=80,25
  X=randn(rng,n,p); b=randn(rng,n)
  D=spdiagm(p-2,p,0=>ones(p-2),1=>-2ones(p-2),2=>ones(p-2))
  λ,κ=.01,.1
  m=JuMP.Model(HiGHS.Optimizer);JuMP.set_silent(m)
  JuMP.@variable(m,β[1:p]);JuMP.@variable(m,t[1:p]);JuMP.@variable(m,v[1:p-2])
  JuMP.@constraint(m,t .>= β);JuMP.@constraint(m,t .>= -β)
  JuMP.@constraint(m,v .>= D*β);JuMP.@constraint(m,v .>= -D*β)
  JuMP.@objective(m,Min,sum((X*β-b).^2)/(2n)+λ*sum(t)+κ*sum(v))
  JuMP.optimize!(m);@test JuMP.is_solved_and_feasible(m)
  res=solve!(zeros(p),Scaled(LeastSquaresLoss(b,X),1/n),NormL1(λ),NormL1(κ),D,
             CondatVu(abstol=1e-10,reltol=1e-9),ProximalOptions(maxiter=100_000))
  @test res.converged
  @test abs(res.minimum-JuMP.objective_value(m)) < 1e-8
  @test norm(res.minimizer-JuMP.value.(β),Inf) < 2e-6
  @printf("GENERALIZED_LASSO iterations=%d ΔF=%.3e Δx=%.3e\n",res.iterations,res.minimum-JuMP.objective_value(m),norm(res.minimizer-JuMP.value.(β),Inf))
end
ORACLE_ONLY && exit()

open(joinpath(OUT,"comparison.csv"),"w") do io
  println(io,"problem,n,p,method,seconds,iterations,objective,relative_error,converged,primal_error,dual_error,bytes_per_iteration")
  for (n,p) in ((500,100),(2000,200))
    q=.5; X,b,λ=quantile_data(n,p,q)
    xlp,ylp,Fref=quantile_lp(X,b,q,λ)
    f,g,h=ZeroFunction(),NormL1(λ),Scaled(Translate(CheckLoss(1-q),b),1/n)
    nd=opnorm(X)
    # Warm compilation before all timed runs.
    solve!(zeros(p),f,g,h,X,CondatVu(normD=nd),ProximalOptions(maxiter=2))
    run_pa_iters(PA.ChambollePock,2;x0=zeros(p),y0=zeros(n),g=PAProx(g),h=PAProx(h),L=X,gamma=(.99/nd,.99/nd))
    for (name,adapt,restart) in (("native fixed",false,false),("native adaptive",true,true))
      tm=@elapsed res=solve!(zeros(p),f,g,h,X,CondatVu(normD=nd,abstol=1e-8,reltol=1e-7,adaptive=adapt,restart=restart),
                            ProximalOptions(maxiter=1_000_000,gaptol=1e-6,gapevery=25,time_limit=MAXTIME))
      st=PS._pd_state(zeros(p),zeros(n),f,X); hc=Conjugate(h)
      step()=PS._pd_step!(st,f,g,hc,X,.99/nd,.99/nd,1e-8,1e-7)
      step();step(); bytes=@allocated step()
      println(io,join(("quantile",n,p,name,tm,res.iterations,res.minimum,abs(res.minimum-Fref)/abs(Fref),res.converged,norm(res.minimizer-xlp,Inf),norm(res.y-ylp,Inf),bytes),',')); flush(io)
    end
    tm=@elapsed sol=pa_gap(PA.ChambollePock,f,g,h,X,(.99/nd,.99/nd),1e-6)
    xx,yy=sol[1]; F=g(xx)+h(X*xx)
    b1=@allocated run_pa_iters(PA.ChambollePock,100;x0=zeros(p),y0=zeros(n),g=PAProx(g),h=PAProx(h),L=X,gamma=(.99/nd,.99/nd))
    b2=@allocated run_pa_iters(PA.ChambollePock,200;x0=zeros(p),y0=zeros(n),g=PAProx(g),h=PAProx(h),L=X,gamma=(.99/nd,.99/nd))
    println(io,join(("quantile",n,p,"PA ChambollePock",tm,sol[2],F,abs(F-Fref)/abs(Fref),duality_gap(f,g,h,X,xx,yy).gap/max(abs(F),floatmin(Float64))<=1e-6,norm(xx-xlp,Inf),norm(yy-ylp,Inf),(b2-b1)/100),','));flush(io)
    tm=@elapsed quantile_lp(X,b,q,λ)
    println(io,join(("quantile",n,p,"HiGHS LP",tm,0,Fref,0,true,0,0,0),','));flush(io)
  end
  for n in (1000,10000)
    rng=MersenneTwister(23+n); b=repeat(randn(rng,10);inner=n÷10)+.5randn(rng,n)
    D=spdiagm(n-1,n,0=>-ones(n-1),1=>ones(n-1))
    f,g,h=SqrNormL2(1.,b),ZeroFunction(),NormL1(1.)
    truth=prox(TotalVariation1D(1.),b); Fref=f(truth)+h(D*truth)
    solve!(zeros(n),f,g,h,D,CondatVu(),ProximalOptions(maxiter=2))
    run_pa_iters(PA.VuCondat,2;x0=zeros(n),y0=zeros(n-1),f=PASmooth(f),beta_f=1.,h=PAProx(h),L=D,gamma=(.4,.49))
    for (name,adapt,restart) in (("native fixed",false,false),("native adaptive",true,true))
      tm=@elapsed res=solve!(zeros(n),f,g,h,D,CondatVu(abstol=1e-8,reltol=1e-7,adaptive=adapt,restart=restart),
                            ProximalOptions(maxiter=1_000_000,gaptol=1e-8,gapevery=25,time_limit=MAXTIME))
      st=PS._pd_state(zeros(n),zeros(n-1),f,D);hc=Conjugate(h)
      step()=PS._pd_step!(st,f,g,hc,D,.4,.4,1e-8,1e-7);step();step();bytes=@allocated step()
      println(io,join(("TV",n,n,name,tm,res.iterations,res.minimum,abs(res.minimum-Fref)/abs(Fref),res.converged,norm(res.minimizer-truth,Inf),NaN,bytes),','));flush(io)
    end
    tm=@elapsed sol=pa_gap(PA.VuCondat,f,g,h,D,(.4,.49),1e-8;f=PASmooth(f),beta_f=1.)
    xx,yy=sol[1];F=f(xx)+h(D*xx)
    b1=@allocated run_pa_iters(PA.VuCondat,100;x0=zeros(n),y0=zeros(n-1),f=PASmooth(f),beta_f=1.,h=PAProx(h),L=D,gamma=(.4,.49))
    b2=@allocated run_pa_iters(PA.VuCondat,200;x0=zeros(n),y0=zeros(n-1),f=PASmooth(f),beta_f=1.,h=PAProx(h),L=D,gamma=(.4,.49))
    println(io,join(("TV",n,n,"PA VuCondat",tm,sol[2],F,abs(F-Fref)/abs(Fref),duality_gap(f,g,h,D,xx,yy).gap/max(abs(F),floatmin(Float64))<=1e-8,norm(xx-truth,Inf),NaN,(b2-b1)/100),','));flush(io)
    tm=@elapsed direct=prox(TotalVariation1D(1.),b)
    println(io,join(("TV",n,n,"direct TV prox",tm,0,Fref,0,true,norm(direct-truth,Inf),NaN,0),','));flush(io)
  end
end
println("PRIMAL_DUAL_BENCHMARK_COMPLETE")
