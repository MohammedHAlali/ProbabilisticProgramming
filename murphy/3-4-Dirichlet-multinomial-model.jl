using Distributions
using StatsFuns
using Plots
include("/home/abergman/projects/optimization/SPSA.jl")

# {{{
counts = [2,4,4,0,1,1,0,1,0,4]
size(counts)

type MyModel <: Distribution
  data
  prior
end # type MyModel

model = MyModel( counts, Dirichlet(10,1) )

gp = rand(model.prior) |> zeros

logpdf( model.prior, softmax(gp) )

sum(gp)

logpdf( Multinomial(17,softmax(gp)), model.data )

function Distributions.logpdf( model::MyModel, θ::Vector)
  θ = θ + 1e-8
  θ = θ / sum(θ)
  lp = 0.0
  lp+= logpdf( Multinomial(17,θ), model.data )
  if isnan(lp) 
    copy!(gp,θ)
    println("θ: $θ")
    error("NaN: llh")
  end
  lp+= logpdf( model.prior, θ)
  if isnan(lp)
    copy!(gp,θ)
    println("θ: $θ")
    error("NaN: prior")
  end
  return lp
end # function logjoint

function Distributions.logpdf( model::MyModel, θ::Matrix)
  [ logpdf( model, θ[:,i] ) for i in 1:size(θ)[2] ]
end # function logjoint
logpdf( model, rand(model.prior, 10) )

function SPSAmod.loss( alpha::Vector )
  posterior = Dirichlet( exp(alpha) )
  SPSAmod.kldivergence( posterior, model, 1000)
end
SPSAmod.loss( rand(model.prior) )
#}}}

init = rand(model.prior) |> zeros
spsa = SPSAmod.SPSA( init )
spsa.kd = 1
SPSAmod.clear_history( spsa, true)
SPSAmod.searchADAM( spsa, 2000, 1)
plot(spsa)

dat = hcat(spsa.param_history...)
plot(dat[6,:])

exp(init)

mean( Dirichlet( exp(init) ) )' * 27

[3,5,5,1,2,2,1,2,1,5] |> sum


entropy( model.prior )

alpha = rand(model.prior)

[ SPSAmod.crossentropy( rand(model.prior,100), model) for i in 1:100] |> mean

[ SPSAmod.Entropy(rand(model.prior,100), model ) for i in 1:100] |> mean

[ SPSAmod.Entropy(rand(model.prior,100), Dirichlet(rand(model.prior)) ) for i in
 1:100] |> mean


using TextAnalysis

vocab = ["mary","lamb","little","big","fleece","white","black","snow","rain","unknown"]
w2i = Dict( zip(vocab,1:10) )

txt = StringDocument("mary had a little lamb, little lamb, little lamb,
                      mary had a little lamb, its fleece as white as snow")
prepare!(txt, strip_punctuation|strip_stopwords)
ngrams(txt)

map( x->w2i[x], tokens(txt) )

w2i[tokens(txt)]

w2i["mary"]
