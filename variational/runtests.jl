using Base.Test
include("./variational.jl")

srand(1234)

k = 3
m = 2
mus = zeros(m,k)
zp  = zeros(k)
q = collect(1.:18);



#
# unpack_chol!
println("Testing: unpack_chol!")
m = 3
R = zeros(m,m)
s = 0
Q = 1:m*m |> collect |> float
t = MixtureModels.unpack_chol!(R, s, Q)
R

# @code_warntype MixtureModels.unpack_chol!(R, s, Q)

# @code_llvm MixtureModels.unpack_chol!(R, s, Q)

# @code_native MixtureModels.unpack_chol!(R, s, Q)

# @time MixtureModels.unpack_chol!(R, s, Q)

#
# MixtureModel construction
println("Testing: MixtureModel construction")
model = MixtureModels.MixtureModel(k,m)

#
# unpack!
println("Testing: unpack!")
q = 1:100 |> float |> collect
MixtureModels.unpack!(model,q);

@test model.mus[1] == [1.,2.,3.]
@test model.mus[2] == [4.,5.,6.]
@test model.mus[3] == [7.,8.,9.]

@test (model.R[1] .== model.mvnorms[1].Σ.chol.factors) |> vec |> all
@test (model.R[2] .== model.mvnorms[2].Σ.chol.factors) |> vec |> all
@test (model.R[3] .== model.mvnorms[3].Σ.chol.factors) |> vec |> all

using StatsFuns
@test model.zp == softmax([28,29,30])

#
# negloglikelihood
println("Testing: negloglikelihood")
model = MixtureModels.MixtureModel(3,2)
df_data = MixtureModels.GenerateData();
X = convert(Array,df_data[[:x1,:x2]]);
nll = MixtureModels.Gen_negloglikelihood( model, X)
Q = MixtureModels.init_Q(model, X);

@inferred  nll(Q)

#
# initCov() == unpack_chol!()
println("Testing: initCov() == unpack_chol!()")
m = 2 
S = cov(X)
R = chol(S)
r = MixtureModels.initCov( R )
C = zeros(m,m)
MixtureModels.unpack_chol!( C, 0, r)
@test R == C


