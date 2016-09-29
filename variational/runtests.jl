using Base.Test
include("./variational.jl")

srand(1234)

k = 3
m = 2
r = round(Int64, (m^2+m)/2 )
mus = zeros(m,k)
rig = zeros(r,k)
zp  = zeros(k)
q = collect(1.:18);

MixtureModels.unpack!(mus,rig,zp,q)

@test round(Int,vec(mus)) == collect(1:6)
@test round(Int,vec(rig)) == collect(7:15)
@test round(Int,vec(zp)) == collect(16:18)


@code_warntype MixtureModels.unpack_array!(mus,0,q)

@time MixtureModels.unpack_array!(mus,0,q)

@code_warntype MixtureModels.unpack!(mus,rig,zp,q)

@time MixtureModels.unpack!(mus,rig,zp,q)

#
# genChol!
r = MixtureModels.initCov(4)
X = zeros(4,4)
MixtureModels.genChol!(X, r)
X

@code_warntype MixtureModels.genChol!(X,r)

@code_llvm MixtureModels.genChol!(X,r) 

@time MixtureModels.genChol!(X,r)

#
# gen_distributions!
rig = [ MixtureModels.initCov(m) for i=1:3] 
rig = hcat( rig... )
R = [zeros(m,m) for i=1:k]
@code_warntype MixtureModels.gen_distributions!(mus, rig, R, zp, k)

@code_llvm MixtureModels.gen_distributions!(mus, rig, R, zp, k)

@time MixtureModels.gen_distributions!(mus, rig, R, zp, k);






