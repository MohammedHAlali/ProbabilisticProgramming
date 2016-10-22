
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

using StatsFuns
using Distributions
using DataFrames
using PDMats
using Base.LinAlg.Cholesky
using Base.LinAlg.BLAS

type MixtureModel
  k::Int # number of mixtures
  m::Int # number of features
  mus::Array{Array{Float64,1},1}
  zp::Array{Float64,1}
  R::Array{Array{Float64,2},1}
  RtR::Array{Array{Float64,2},1}
  mvnorms::Array{FullNormal}
  zdist::Categorical
  lH::Array{Float64,1}
  lp_mvn::Array{Array{Float64,1},1}
  lp_z::Array{Float64,1}
  function MixtureModel(k,m)
    model = new(k,m)
    model.mus = [zeros(m) for i=1:k]
    model.zp  = zeros(k)
    model.R = Array{Matrix}(k)
    model.RtR = Array{Matrix}(k)
    model.mvnorms = Array{FullNormal}(k)
    for i = 1:k
      model.mvnorms[i] = MvNormal( model.mus[i], PDMat(eye(m)) )
      model.R[i] = model.mvnorms[i].Σ.chol.factors
      model.RtR[i] = model.mvnorms[i].Σ.mat
    end
    model.zdist = Categorical(k)
    return model
  end
end

function getKM( model::MixtureModel )
  return model.k, model.m
end

function unpack_chol!( R::Array{Float64,2}
                      ,s::Int
                      ,Q::Array{Float64,1} )
  m = size(R,1)
  for j = 1:m
    for i = 1:m
      if i > j
        continue
      end
      s += 1
      R[i,j] = Q[s]
    end
  end
  return s
end

function initCov( U )
  m = size(U,1)
  r = Float64[]
  for j = 1:m
    for i = 1:m
      if i > j
        continue
      end
      push!(r, U[i,j])
    end
  end
  return r
end

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

function unpack_array!( a::AbstractArray
                       ,s::Int
                       ,q::Array{Float64,1} )
  for i in eachindex(a)
    s+=1
    a[i] = q[s]
  end
  return s
end # function f

function unpack!( model::MixtureModel, Q::Vector )
  k = model.k

  s = 0
  for i = 1:k
    s = unpack_array!(model.mus[i],s,Q)
  end

  for i = 1:k
    R = model.R[i]
    s = unpack_chol!(R, s, Q)
    fill!(model.RtR[i],0.0)
    gemm!('T','N',1.0,R,R,1.0,model.RtR[i])
  end
 
  zp = model.zp
  s = unpack_array!(zp, s, Q)
  softmax!(zp, zp)

  return s
end # function unpack!

function Gen_negloglikelihood( model::MixtureModel, X::Array{Float64,2} )
  @assert model.m == size(X,2) 
  k = model.k
  N = size(X,1)
  model.lH = zeros(k)
  model.lp_mvn = [zeros(N) for i=1:k]
  model.lp_z = zeros(k)
  function nll( q::Array{Float64,1} )
    negloglikelihood( model, X, q )
  end
  return nll
end

function negloglikelihood(model::MixtureModel, X::Array{Float64,2}, q::Array{Float64,1} )
  k, m = getKM(model)
  N = size(X,1)
  lp = 0.0
  unpack!( model, q)
  for h = 1:k
    logpdf!(model.lp_mvn[h], model.mvnorms[h], X')
    model.lp_z[h] = log( model.zp[h] )
  end
  @inbounds for n in 1:N
    for h = 1:k
      model.lH[h] = model.lp_mvn[h][n] + model.lp_z[h]
    end
    lp += logsumexp( model.lH )
  end
  return -lp
end # function f

function setparams!( model, spsa, idx::Int64 )
  Q = convert(Array{Float64}, spsa.param_history[idx,:]) |> vec
  unpack!( model, Q)
  nothing
end


function getZp(model::MixtureModel, spsa)
  N = size(spsa.param_history,1)
  zp = zeros(6, N)
  for idx = 1:N
    setparams!( model, spsa, idx)
    zp[:,idx] = [ model.zp ; softmax(model.zp) ]
  end
  df_hist = spsa.param_history[:,[:iter,:M,:simNum]] 
  return [DataFrame(zp') df_hist]
end # function getZp

function getMus(spsa)
  [spsa.param_history[:,[:iter,:M,:simNum]] spsa.param_history[:,1:6]]
end # function getMus


function getCov!(model::MixtureModel, spsa, idx::Int64, T::Int=100 )

  setparams!( model, spsa, idx)

  σ = 2.0
  k = model.k
  t = linspace(0,2*pi,T)
  circle = σ*[cos(t)  sin(t)]'
  ellipse = zeros(2,T,k)

  for t = 1:T
    for h = 1:k
      ellipse[:,t,h] = model.R[h]' * circle[:,t] + model.mus[h]
    end
  end

  return map(1:k) do k
    df = DataFrame( ellipse[:,:,k]' )
    df[:k] = k
    df
  end |> vcat
end

function init_Q( model::MixtureModel, X::Matrix )

  S = cov(X)
  R = chol(S)
  r = initCov( R )

  mu = mean(X,1) |> vec

  mu_init = [ mu ; mu ; mu ]
  s_init  = [  r ;  r ;  r ] 
  z_init  = zeros( model.k )
  q_init = [mu_init ; s_init ; z_init ]
  return q_init
end # function init_Q

function posteriorResponsibility()
  responsibility = Matrix(N,3)

  sum(responsibility,2)

  for k = 1:K
    responsibility[d,k] = pdf(mvnorms[k],x) * pdf(zdist,k)
  end
end

function profile_negloglikelihood()
  model = MixtureModel(3,2)
  df = GenerateData()
  X = convert(Array,df[[:x1,:x2]]) 
  Q = init_Q(model, X) 
  nll = Gen_negloglikelihood( model, X )
  nll( Q )
  Profile.clear_malloc_data()
  sol = 0.0
  for i = 1:1000
    sol += nll(Q)
  end
  print( hash(sol) )
end # function profile_negloglikelihood

end # module MixtureModels }}}


