include("/home/abergman/projects/optimization/SPSA.jl")

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

