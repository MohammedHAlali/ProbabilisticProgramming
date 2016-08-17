module ParticleFilter #{{{

using Distributions
using DataFrames

srand(1)

T = 100
state = zeros(T)
obs = zeros(state)

state0 = 10.
wdist = Normal(0,1)
vdist = Normal(0,2)

state[1] = state0
for t = 2:T
  state[t] = state[t-1] + rand(wdist)
  obs[t] = state[t] + rand(vdist)
end

dfData = DataFrame( t=1:T, state=state, obs=obs)

function EffectiveSampleSize( weights::Array{Float64,1} )
  1/sum(weights.^2)
end

function proposalSample( state0 )
  state1 = state0 + rand(wdist)
  return state1
end

function proposalDensity(state0, state1)
  w = state1 - state0
  pdf(wdist, w)
end

function posteriorDensity(obs, state0, state1)
  pdf(vdist, obs - state1) * pdf(wdist, state1 - state0)
end

function likelihood( obs, state )
  pdf(vdist, obs - state)
end



N = 100
particles = fill(state0,T,N)
weights   = fill(  1./N,T,N)

for t = 2:T
  for n = 1:N
    particles[t,n] = proposalSample( particles[t-1,n] )
    weights[t,n] = likelihood( obs[t], particles[t,n] )
  end
  q = weights[t,:] / sum(weights[t,:]) |> vec
  idx = rand( Categorical(q), N)
  particles[t,:] = particles[t,idx]
end

function SIR()
    weights[t,n] = weights[t-1,n] * 
      posteriorDensity( obs[t], particles[t-1,n], particles[t,n] ) /
      proposalDensity( particles[t-1,n], particles[t,n] )
end


df = DataFrame( Dict( [( n, particles[:,n]) for n in 1:N]))
df = hcat( df, DataFrame(t=1:T))
df = stack(df)
df[:variable] = map(string,df[:variable])


# describe(df)

# dump(df)

end # ParticleFilter }}}

