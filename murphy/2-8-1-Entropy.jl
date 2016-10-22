using Plots
using StatPlots
using Distributions
include("/home/abergman/projects/optimization/SPSA.jl")

entropy( Categorical([0.25, 0.25, 0.2, 0.15, 0.15]), 2 )

dir = Dirichlet(4, 1e-0)
init = rand(dir)

model = Categorical(init)
entropy( model )

function myentropy( dist::Categorical )
  sum = 0.0
  for p in params(dist)[1]
    sum += p * log(2, p)
  end
  return -sum
end

myentropy( model )

entropy( model, 2 )


for p in params( model )[1]
  println("$p")
end

# maximize entropy yields uniform distribution
function SPSAmod.loss( model::Categorical )
  return -myentropy( model )
end

spsa = SPSAmod.SPSA( model )

SPSAmod.searchADAM( spsa , 1000 )
SPSAmod.plot(spsa)

bar(probs(model))

model

myentropy( model )


spsa.param_history

spsa.loss_history
