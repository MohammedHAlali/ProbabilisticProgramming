using DataFrames
using RCall
using Distributions
using ProfileView
using StatsFuns
using Base.Test

include("variational.jl")
include("/home/abergman/projects/optimization/SPSA.jl")

# Running MixtureModels {{{
df_data = MixtureModels.GenerateData();
@rput df_data
X = convert(Array,df_data[[:x1,:x2]]);

k = 3
m = 2
model = MixtureModels.MixtureModel(k,m)
Q = MixtureModels.init_Q(model, X);
negloglikelihood = MixtureModels.Gen_negloglikelihood(model, X)


spsa = SPSAmod.SPSA( Q, negloglikelihood)
SPSAmod.clear_history(spsa,true)
spsa.dQmax = 0.001
@time SPSAmod.searchADAM( spsa, 10000, 5)

df_mus = MixtureModels.getMus(spsa)
@rput df_mus
df_cov = MixtureModels.getCov!( model, spsa, 1 );
@rput df_cov

spsa.param_history

#}}}


include("variational.jl")
MixtureModels.profile_negloglikelihood()

Profile.clear()
spsa = SPSAmod.SPSA( Q, negloglikelihood)
SPSAmod.clear_history(spsa, true)
spsa.dQmax = 0.001
@profile SPSAmod.searchADAM( spsa, 10000 )
s = open("/tmp/prof.txt","w")
# Profile.print(s, format=:flat, sortedby=:count )
Profile.print(s)
close(s)
ProfileView.view()


#
# negloglikelihood
include("variational.jl")
k = 3
m = 2
model = MixtureModels.MixtureModel(k,m)
Q = MixtureModels.init_Q(model, X);
negloglikelihood = MixtureModels.Gen_negloglikelihood(model, X)
println( negloglikelihood(Q) )
@time MixtureModels.negloglikelihood(model, X, Q)

@code_warntype negloglikelihood(Q)
@code_warntype MixtureModels.Gen_negloglikelihood(model, X)

@time negloglikelihood(Q)

@code_warntype MixtureModels.negloglikelihood(model, X, Q)
@time MixtureModels.negloglikelihood(model, X, Q)

Profile.clear()
@profile for i=1:1000 negloglikelihood(Q) end
ProfileView.view()

@code_warntype logsumexp(Q)

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




#
# getZp()
k = 3
m = 2
include("./variational.jl")
model = MixtureModels.MixtureModel(k,m)
spsa = SPSAmod.SPSA( Q, negloglikelihood)
SPSAmod.clear_history(spsa,true)
spsa.dQmax = 0.0001
@time SPSAmod.searchADAM( spsa, 100, 1)
df_zp=MixtureModels.getZp(model, spsa)




@code_warntype MixtureModels.unpack!(model, q)

@code_native MixtureModels.unpack!(model, q)


include("./variational.jl")
MixtureModels.getCov( model, spsa, 3)



include("/home/abergman/projects/optimization/SPSA.jl")
spsa = SPSAmod.SPSA( Q, negloglikelihood)
grad = zeros(spsa.N)
@code_warntype SPSAmod.SPSAgrad!(spsa, grad, Q, 1e-3, 1)

SPSAmod.SPSAgrad!(spsa, grad, Q, 1e-3, 1)
@time SPSAmod.SPSAgrad!(spsa, grad, Q, 1e-3, 1)

  0.002681 seconds (93 allocations: 34.641 KB)
248.69427209199827

  0.000254 seconds (91 allocations: 34.609 KB)
192.03294180721818

@time spsa.lossFunction( Q )

@time negloglikelihood( Q )

include("./memory-profile.jl")

h=1
@time sol=logpdf!(model.lp_mvn[h], model.mvnorms[h], X');

Pkg.update()
Pkg.add("Coverage")

using Coverage

v = analyze_malloc(".")

v = analyze_malloc("/home/abergman/.julia/v0.5/Distributions")

v = analyze_malloc("/home/abergman/projects/optimization")

v[1]
