using Distributions
using StatsFuns
using Plots
using BenchmarkTools
using DataFrames
include("/home/abergman/projects/optimization/SPSA.jl")

immutable LogisticRegression
  w::Vector{Float64}
  function LogisticRegression(d::Int)
    new( zeros(d+1) )
  end
end

function sigm(dist::LogisticRegression, x::Matrix)
  X = hcat( ones(x[:,1]) , x )
  μ = logistic.(X * dist.w)
  return μ
end

function logpdf(dist::LogisticRegression, y::Vector, x::Matrix )
  μ = sigm( dist, x)
  ϵ = realmin()
  y.*log(ϵ+μ) + (1-y).*log(ϵ+(1-μ))
end

function Distributions.rand( dist::LogisticRegression, x::Matrix)
  [ rand(Bernoulli(p)) for p in sigm(dist, x) ]
end

n = 100
d = 40
dist = LogisticRegression(d)
splice!(dist.w,1:(d+1),randn(d+1))
const x = rand(Uniform(-8,8), n, d)
const y = rand(dist, x)

dist.w

logpdf( dist, y, x ) |> mean

function Base.append!(Q::Vector, m::LogisticRegression)
  append!(Q, m.w)
end # function length

function SPSAmod.unpack!( dist::LogisticRegression, s::Int, Q::Vector )
  SPSAmod.unpack!( dist.w, s, Q)
end

function SPSAmod.loss( model::LogisticRegression )
  lp = logpdf( model, y, x) |> mean
  return -lp
end
SPSAmod.loss(dist)

model = LogisticRegression(d)
spsa = SPSAmod.SPSA( model )
SPSAmod.clear_history(spsa, true)
SPSAmod.searchADAM( spsa, 1000, 1)
println(model.w)
println(dist.w)
plot(spsa)


dat = hcat(spsa.param_history...)
plot()
[ plot!(dat[i,:]) for i in 1:size(dat,1)]
gui()

df = DataFrame(x)
df[:y] = y
scatter(df,:x1,:x2,:y, m=14)

test_n = 1000
test_x = rand(Uniform(-8,8), test_n, d)
test_y = exp(logpdf( model, ones(test_n), test_x))
scatter!(test_x[:,1], test_x[:,2], test_y)




@benchmark map(p -> rand(Bernoulli(p)) , logistic.( w' * X' ) )

@benchmark [ rand(Bernoulli(p)) for p in logistic.( w' * X' ) ]


