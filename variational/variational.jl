
module variational #{{{

using Distributions
using DataFrames

function lossFunction(q::Array{Float64,1})
  mu_mean, mu_log_sd = q
  Pmu= Normal(20,exp(3))
  Qmu= Normal(10*mu_mean,exp(mu_log_sd))
  Qe = entropy(Qmu)
  M=100
  elbo = Qe + sum(logpdf(Pmu,rand(Qmu,M)))/M
  return -elbo
end # function lossFunction

# M = 10000
# std(Float64[sum(logpdf(Normal(0,exp(3)),rand(Normal(0,exp(3)),M)))/M for i=1:100])


const N = 50
const true_mu = 1
const true_sd = 2
const X = rand(Normal(true_mu,true_sd), N);

function logjoint(mu, sd)
  ll = loglikelihood( Normal(mu,sd), X)
  prior_mu = logpdf(Normal(0,10), mu)
  prior_sd = logpdf(Normal(0,10), sd)
  return ll + prior_mu + prior_sd
end # function logjoint

function evidenceLowerBound(q::Array{Float64,1})
  M = 100
  mu_mean, mu_sd, sd_shape, sd_scale = q
  Qmu = Normal(mu_mean, exp(mu_sd))
  Qsd = Gamma(exp(sd_shape), exp(sd_scale))
  mu = rand(Qmu,M)
  sd = rand(Qsd, M)
  sum = 0
  for i = 1:M
    sum += logjoint( mu[i], sd[i] )
  end
  return sum / M + entropy(Qmu) + entropy(Qsd)
end # function evidenceLowerBound

# elbo = map( r->evidenceLowerBound([1.,2.,8.,8./10]), 1:80)
# std(elbo)

function elbo()

  df=DataFrame(mu=Float64[], elbo=Float64[])
  for mu=-2:0.01:3
    push!(df, (mu, evidenceLowerBound([mu, 1, 3, 8/10]) ) )
  end

  return df
  
end # function elbo



function entropyCalculations( Q)
  fQmu = x -> pdf(Qmu,x) * logpdf(Qmu,x)

  Qsd = Gamma(8, 100/10)

# entropy thru expected value integral
  fQsd = x -> -1 * pdf(Qsd,x) * logpdf(Qsd,x)
  (sol, err) = quadgk(fQsd,0,Inf)

# analytic entropy eval
  entropy( Qsd )

# entropy thru sampling
  M = 10000
  -logpdf(Qsd, rand(Qsd,M))/M |> sum

end # function entropy


function posteriorPDF()
  df = DataFrame(mu=Float64[], sd=Float64[], lp=Float64[])
  for mu = -3:0.1:3
    for sd = 0.1:0.1:5
      push!(df, (mu,sd,logjoint(mu, sd)))
    end
  end
  return df
end

function posteriorVariationalPDF(q)
  mu_mean, mu_sd, sd_shape, sd_scale = q
  Qmu = Normal(mu_mean, exp(mu_sd) )
  Qsd = Gamma( exp(sd_shape), exp(sd_scale) )
  df = DataFrame(mu=Float64[], sd=Float64[], lp=Float64[])
  for mu = -3:0.1:3
    for sd = 0.1:0.1:5
      push!(df, (mu,sd,logpdf(Qmu,mu)+logpdf(Qsd,sd)))
    end
  end
  return df
end

# s = SPSA( [0.,0., 0., 0.], q -> -evidenceLowerBound(q) )

# SPSAsearchRestart(s, 4)


end # module variational }}}

module MixtureModels #{{{

include("/home/abergman/projects/optimization/SPSA.jl")

using StatsFuns
using Distributions
using DataFrames
using PDMats
using Base.LinAlg.Cholesky

function genChol!( R::Array{Float64,2}
                  ,r::Array{Float64,1} )
  m = size(R,1)
  k=0
  for j = 1:m
    for i = 1:m
      if i > j
        continue
      end
      k += 1
      @inbounds R[i,j] = r[k]
    end
  end
  return R
end

function genCov( r::Array{Float64,1} )
  l = length(r)
  m = round(Int64,(sqrt(1+8l)-1)/2)
  k=0
  R=Float64[i<=j?(k+=1;r[k]):0 for i=1:m, j=1:m]
  S=R'*R
end

function randCov( n )
  rand(round(Int64,(n^2+n)/2))-0.5 |> genCov
end

function initCov(m)
  r = Float64[]
  for i = 1:m
    for j = 1:m
      if i==j
        push!(r, 1.)
      elseif i>j
        push!(r, 0.)
      end
    end
  end
  return r
end # function initCov

function GenerateData()
  zp = [0.3,0.5,0.2]
  zdist = Categorical(zp)

  sdist = InverseWishart(5, 0.1*eye(2))

  mvnorms = [ 
      MvNormal([1.,0.], rand(sdist) )
    , MvNormal([2.,3.], rand(sdist) )
    , MvNormal([2.,1.], rand(sdist) )
  ]

  N = 100
  df = DataFrame(z=Int64[], x1=Float64[], x2=Float64[])
  for i in 1:N
    z = rand(zdist)
    x = rand(mvnorms[z])
    push!(df, (z, x...))
  end

  return df
end

function posteriorResponsibility()
  responsibility = Matrix(N,3)

  sum(responsibility,2)

  for k = 1:K
    responsibility[d,k] = pdf(mvnorms[k],x) * pdf(zdist,k)
  end
end

function unpack!( mus::Array{Float64,2}
                 ,rig::Array{Float64,2}
                 , zp::Array{Float64,1}
                 ,  q::Array{Float64,1} )
  s = 0
  s = unpack_array!(mus,s,q)
  s = unpack_array!(rig,s,q)
  s = unpack_array!(zp,s,q)
end # function

function unpack_array!( a::AbstractArray
                       ,s::Int
                       ,q::Array{Float64,1} )
  for i in eachindex(a)
    s+=1
    a[i] = q[s]
  end
  return s
end # function f

function gen_distributions!(mus,rig,R,zp,k)
  mvnorms = map( i->MvNormal(
    mus[:,i],
    PDMat( Cholesky(Matrix(genChol!(R[i],rig[:,i])),'U') ) )
                , 1:k )
  zdist = Categorical(softmax(zp))
  return (mvnorms, zdist)
end # function gen_distributions

function Gen_negloglikelihood( X::Array{Float64,2}, k::Int, m::Int, r::Int )
  lH = zeros(k)
  N = size(X,1)
  lp_mvn = [zeros(N) for i=1:k]
  R = [zeros(m,m) for i=1:k]
  lp_z = zeros(k)
  mus = zeros(m,k)
  rig = zeros(r,k)
  zp  = zeros(k)
  function negloglikelihood( q::Array{Float64,1} )
    unpack!(mus,rig,zp,q)
    (mvnorms, zdist) = gen_distributions!(mus,rig,R,zp,k)
    for h = 1:k
      logpdf!(lp_mvn[h], mvnorms[h], X')
    end
    logpdf!(lp_z, zdist, 1:k)
    lp = 0.0
    @inbounds for n in 1:N
      for h = 1:k
        lH[h] = lp_mvn[h][n] + lp_z[h]
      end
      lp += logsumexp( lH )
    end
    return -lp
  end
  return negloglikelihood
end

function getParams( idx::Int64 )
  (mus, rig, zp) = convert(Array{Float64}, spsa.param_history[idx,:]) |> vec |> unpack
end

function getZp()
  df_zp = Float64[
    begin
      (mus, rig, zp) = getParams(i)
      Float64[zp ; softmax(zp)][j]
    end
    for i = 1:size(spsa.param_history,1), j=1:6 ] |> DataFrame
  df_hist = spsa.param_history[:,[:iter,:M,:simNum]] 
  return [df_zp df_hist]
end # function getZp

function getMus(spsa)
  [spsa.param_history[:,[:iter,:M,:simNum]] spsa.param_history[:,1:6]]
end # function getMus

function getCov(idx::Int64, z::Int64)

  t = linspace(0,2*pi,100) |> collect

  (mus, rig, zp) = getParams(idx)
  cov = genCov(rig[:,z])
  (eVal, eVec) = eig(cov)
 
  c = 1
  a = sqrt( c*eVal )
  X = mus[:,z] .+ ((a.*[cos(t) sin(t)]')'*eVec)'
  X = mus[:,z] .+ ((a.*[cos(t) sin(t)]')'*eVec)'
  
  df = DataFrame(X')
  df[:z] = z
  
  return df
end

function init_Q(k,m,r, df)

  std_init = max(std(df[:x1]),std(df[:x2]))

  mu_init = Float64[[mean(df[:x1]),mean(df[:x2])][a] for a=1:m, i=1:k] |> vec
  s_init  = std_init*[ initCov( m ) ; initCov( m ) ; initCov( m ) ] 
  z_init  = zeros( k )
  q_init = [mu_init ; s_init ; z_init ]
  return q_init
end # function init_Q

function profile_negloglikelihood()
  k = 3
  m = 2
  r = round(Int64, (m^2+m)/2 )
  df = GenerateData()
  X = convert(Array,df[[:x1,:x2]]) 
  Q = init_Q(k,m,r, df) 
  nll = Gen_negloglikelihood( X, k, m, r )
  nll( Q )
  Profile.clear_malloc_data()
  sol = 0.0
  @time for i = 1:1000
    sol = nll(Q)
  end
  print( hash(sol) )
end # function profile_negloglikelihood

end # module MixtureModels }}}


