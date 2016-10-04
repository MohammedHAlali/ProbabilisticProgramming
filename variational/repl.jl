using RCall
using Distributions
using ProfileView
using StatsFuns

include("variational.jl")
include("/home/abergman/projects/optimization/SPSA.jl")

MixtureModels.profile_negloglikelihood()

df_data = MixtureModels.GenerateData();
@rput df_data
X = convert(Array,df_data[[:x1,:x2]]);

k = 3
m = 2
model = MixtureModels.MixtureModel(k,m)
Q = MixtureModels.init_Q(model, df_data);
negloglikelihood = MixtureModels.Gen_negloglikelihood(model, X)

spsa = SPSAmod.SPSA( Q, negloglikelihood)
SPSAmod.clear_history(spsa, true)

model = MixtureModels.MixtureModel(k,m)
MixtureModels.getParams( model, spsa, 1)


spsa.dQmax = 0.01
@time SPSAmod.searchADAM( spsa, 1000, 1)
df_mus = MixtureModels.getMus(spsa)
@rput df_mus

MixtureModels.getCov(1,1)


Profile.clear()
spsa = MixtureModels.SPSAmod.SPSA( Q, negloglikelihood)
@profile MixtureModels.SPSAmod.searchADAM( spsa, 100 )
s = open("/tmp/prof.txt","w")
# Profile.print(s, format=:flat, sortedby=:count )
Profile.print(s)
close(s)
ProfileView.view()

Profile.clear()
@profile MvNormal(pdmat)
ProfileView.view()

spsa.dQmax = 0.001
SPSAmod.searchADAM( spsa, 100 )
spsa.dQmax = 0.0001
SPSAmod.searchADAM( spsa, 100 )

using RCall

MixtureModels.initCov(3)



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


using Gallium
include(Pkg.dir("Gallium/examples/testprograms/misc.jl"))
Gallium.breakpoint(sinthesin,Tuple{Int64})
inaloop(2)


Pkg.add("Debug")
using Debug



k = 3
mus = [zeros(m) for i=1:k]
mus[2]

mvnorms = Array{MvNormal}(3)
mvnorms[1] = MvNormal(mus[1],PDMat(eye(m)))
mvnorms

k = 3
m = 2
include("./variational.jl")
model = MixtureModels.MixtureModel(k,m)

zp = softmax(1:3)
zdist =  Categorical(zp)
println(zdist)

zp[1] = 0.3333333333333333333333333333333333
zp[2] = 0.333
zp[3] = 0.33333
println(zdist)

rand(zdist,100)'

