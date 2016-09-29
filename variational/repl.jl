using RCall
using Distributions

include("variational.jl")

MixtureModels.profile_negloglikelihood()

k = 3
m = 2
r = round(Int64, (m^2+m)/2 )
df = MixtureModels.GenerateData();
X = convert(Array,df[[:x1,:x2]]);
Q = MixtureModels.init_Q(k,m,r, df);

negloglikelihood = MixtureModels.Gen_negloglikelihood(X,k,m,r)

Profile.clear_malloc_data()
@time negloglikelihood(Q)

@code_warntype negloglikelihood(Q)

spsa = MixtureModels.SPSAmod.SPSA( Q, negloglikelihood)
MixtureModels.SPSAmod.clear_history(spsa, true)

spsa.dQmax = 0.01
@time MixtureModels.SPSAmod.searchADAM( spsa, 100 )

@time MixtureModels.SPSAmod.searchADAM( spsa, 1000 )

@time SPSAmod.SPSAgrad(spsa, q_init, 0.001)

df_mus = MixtureModels.getMus(spsa)
@rput df_mus

R"""
library(ggplot2)
library(dplyr)
library(tidyr)
"""

R"""
df_mus <- df_mus %>% tbl_df() %>% 
  mutate( simNum = ordered(simNum) ) %>%
  rename( x1.1 = x1
         ,x2.1 = x2
         ,x1.2 = x3
         ,x2.2 = x4
         ,x1.3 = x5
         ,x2.3 = x6
         ) %>% 
  gather( param, value, -iter, -simNum, -M ) %>% 
  separate(param, c("v1","z")) %>% 
  spread( v1, value)
"""

R"""
qplot(x=iter,y=x1,data=df_mus)
"""

using ProfileView

Profile.clear()
spsa = MixtureModels.SPSAmod.SPSA( Q, negloglikelihood)
@profile MixtureModels.SPSAmod.searchADAM( spsa, 100 )
s = open("/tmp/prof.txt","w")
# Profile.print(s, format=:flat, sortedby=:count )
Profile.print(s)
close(s)
ProfileView.view()

spsa.dQmax = 0.001
SPSAmod.searchADAM( spsa, 100 )
spsa.dQmax = 0.0001
SPSAmod.searchADAM( spsa, 100 )

using RCall

MixtureModels.initCov(3)


rig = zeros(r,k)
q = 1:(r*k) 
j = 0
for i in eachindex(rig)
  j += 1
  rig[i] = q[j]
end

df = DataFrame([Float64, Int], [:x,:t], 10)

include("./runtests.jl")

using Lint

lintfile("./variational.jl")


# timing logpdf 
iw = InverseWishart(4,0.1*eye(2));
dist =  MvNormal([2.,5.],rand(iw));
N = 10000
data = rand(dist, N)
@time sol=logpdf( dist, data);
@time logpdf!(sol,dist,data);
@time for i = 1:N
  sol[i] = logpdf(dist,data[:,i])
end

logpdf(dist, X')



function f(x::Float64)
  grad = zeros(10)
  grad[3] = x
  return grad
end # function f

function for_f(x::Float64)
  for i::Float64 = 1:1000
    f(i)
  end
end

function gen_h()
  grad = zeros(10)
  function h(x::Float64)
    grad[3] = x
    return grad
  end
end

function for_h(h::Function)
  for i::Float64 = 1:1000
    h(i)
  end
end # function for_h


function m!(grad::Array{Float64,1}, x::Float64)
  grad[3] = x
end

function for_m()
  grad = zeros(Float64,10)
  for i::Float64 = 1:1000
    m!( grad, i )
  end
  return grad
end

f(1.)
for_f(1.)
for_h(f)
for_m()
  


h = gen_h()
# @time f(3.)
# @time for_f(3.)
# @time for_h(f)
# @time for_h(h)
# @time for_m()

using Gallium
include(Pkg.dir("Gallium/examples/testprograms/misc.jl"))
Gallium.breakpoint(sinthesin,Tuple{Int64})
inaloop(2)

Union{
      Array{
            Distributions.MvNormal{
                                   Float64
                                   ,PDMats.PDMat{
                                                 Float64
                                                 ,Array{Float64,2}
                                                }
                                   ,Array{Float64,1}
                                  }
            ,1
           }
      ,Distributions.Categorical
     }
