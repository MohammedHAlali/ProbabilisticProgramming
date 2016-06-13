module Ex34MonteCarloIntegration #{{{

using DataFrames

function h(x)
  ( cos( 50*x ) + sin( 20*x ) ).^ 2
end

M = 10000

x = linspace(0,1,M)

u = rand(M)

df = DataFrame(
    m=1:M
  , x=x
  , h=h(x)
  , u=u
  , hu=h(u)
)

df[:hm] = cumsum(df[:hu]) ./ df[:m]

function variance_m(m::Int64)
  sum((df[:hu][1:m] - df[:hm][m]).^2) / m
end

df[:v] = map( variance_m, 1:M)

df[:se] = sqrt(df[:v]) ./ sqrt(df[:m])

# online variance calc (wikipedia)
df[:var] = zeros(M)
df[:mean] = zeros(M)
df[:mean][1] = df[:hu][1]

M2 = 0
for m = 2:M
  delta = df[:hu][m] - df[:mean][m-1]
  df[:mean][m] = df[:mean][m-1] + delta / m
  M2 += delta * (df[:hu][m] - df[:mean][m])
  df[:var][m] =  M2 / m
end

df[:se2] = sqrt(df[:var]) ./ sqrt(df[:m])

df[[:hm,:mean]]

df[[:v,:var]]

end # Ex34MonteCarloIntegration }}}


module iSampling #{{{
using Distributions
using DataFrames

ndist = Normal(0,100)

r = rand( ndist, 100 )

df = DataFrame(r=r)




end # iSampling }}}

