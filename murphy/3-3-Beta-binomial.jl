using Distributions
using Plots
include("/home/abergman/projects/optimization/SPSA.jl")

dist = Bernoulli( 0.9 )

histogram(rand(dist,100))

dist = Binomial( 20, 0.1)

dist.p = 1

rand(dist,100) |> histogram

prior = Beta(2,2)

plot(x->logpdf(prior,x), 0, 1, ylim=[-3,0.5])


function Distributions.logpdf( p::Matrix, q::Vector )
  -joint.(q)
end

logpdf( Matrix(zeros(1,1)), [0.1,0.2,0.3] )

function joint( θ::Float64 )
  dist = Binomial( 20, θ )
  logpdf(dist, 3) + logpdf(prior, θ)
end
joint.([0.4,0.1])

plot( x->joint([x]), 0, 1, ylim=[-8,-1] )

plot( x->exp(joint([x])), 0, 1 )

function ELBO( params::Vector )

end

function crossentropy(q_samples::Vector, p )
  return logpdf(p, q_samples) |> mean
end

function KL(q, p)
  samples_q = rand(q, 100)
  crossentropy(samples_q, p) - entropy(q)
end

function SPSAmod.loss( params::Vector )
  qdist = Beta( exp(params)... )
  return KL( qdist, zeros(3,3) )
end
SPSAmod.loss( log([5,19]) )

init = ones(2)
spsa = SPSAmod.SPSA( init )
spsa.c = 0.1
spsa.kd = 100
SPSAmod.clear_history(spsa,true)
SPSAmod.searchADAM( spsa, 3000, 1 )
plot(spsa)

spsa.param_history 

spsa.loss_history 

posterior = Beta(exp(init)...)
plot(x->pdf(posterior,x),0,1,ylim=[0,10])

rng = linspace(1,length(spsa.param_history),10)
rng = linspace(1,length(spsa.param_history)/25,10)
plot()
for i in round(Int, rng)
  posterior = Beta(exp(spsa.param_history[i])...)
  plot!(x->pdf(posterior,x),0,1,ylim=[0,10],show=true)
  sleep(0.01)
end

entropy( posterior )

entropy( Beta(exp([1,1])...) )

SPSAmod.loss( init )

dat = hcat(spsa.param_history...)
plot( dat[2,:] )

