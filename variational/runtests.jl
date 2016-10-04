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

rig = Array(Array, 3)
rig[1] = zeros( r )
rig[2] = zeros( r )
rig[3] = zeros( r )
rig::Array{Array,1}

for i in eachindex(rig)
  for j in eachindex(rig[i])
    print( rig[i][j] )
  end
end

#
# unpack_chol!
m = 3
R = zeros(m,m)
s = 0
Q = 1:m*m |> collect |> float
t = MixtureModels.unpack_chol!(R, s, Q)
R

@code_warntype MixtureModels.unpack_chol!(R, s, Q)

@code_llvm MixtureModels.unpack_chol!(R, s, Q)

@code_native MixtureModels.unpack_chol!(R, s, Q)

@time MixtureModels.unpack_chol!(R, s, Q)


#
# unpack!
rig = [ MixtureModels.initCov(m) for i=1:3] 
rig = hcat( rig... )
R = [zeros(m,m) for i=1:k]

include("./variational.jl")
model = MixtureModels.MixtureModel(k,m)
@time MixtureModels.unpack!(model,q);


using StatsFuns

zp = 1.0:10000.0 |> collect
@time zdist = Categorical(softmax!(zp, zp) );
@time zdist = Categorical(softmax!(zp, zp), Distributions.NoArgCheck());

m = 1000
r = round(Int64, (m^2+m)/2 )
rig = ones(r)
R = zeros(m,m)
@time MixtureModels.genChol!(R,rig)
@time Matrix(R)
@time Cholesky( Matrix(R) ,'U') 

chol = Cholesky( Matrix(R) ,'U') 
@time PDMat( chol )

@time PDMat( Cholesky( Matrix(R) ,'U') )

pdmat = PDMat( chol )

m = 2
r = rand(m,m)
pdmat = PDMat(r'r)

mus = ones(m)
@time MvNormal( pdmat )
@time MvNormal( mus, pdmat )

R = Float64[2. 1; 0 1]
pdmat = PDMat( Cholesky( Matrix(R) ,'U') )
mus = ones(2)
mvn = MvNormal( mus, pdmat )

mus[2] = 123.
mvn

L = mvn.Σ.chol.factors
fill!(mvn.Σ.mat, 0.0)
gemm!('T','N',1.0,L,L,1.0, mvn.Σ.mat)
mvn

mvn.Σ.mat

L'*L

L[1] = -3



pdmat.chol

pdmat.chol |> fieldnames

pdmat.chol.factors

pdmat.chol[1] = 123.
pdmat.chol

pdmat.mat = full(pdmat.chol)

using Base.LinAlg.BLAS

m=2
mat = zeros(m,m)

R'R

full( Cholesky(R, 'U') )



using PDMats
using Base.LinAlg.Cholesky

mvnorms = map(1:k) do i
  MvNormal( mus[:,i] 
           , PDMat( Cholesky( Matrix( MixtureModels.genChol!(R[i],rig[:,i])) ,'U') ) )
end

@code_warntype MixtureModels.unpack!(model, q)



MixtureModels.Gen_negloglikelihood( model, X )

@time negloglikelihood(Q)

@code_warntype negloglikelihood(Q)
