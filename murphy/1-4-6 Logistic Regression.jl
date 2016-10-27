using Distributions
using StatsFuns
using Plots
using BenchmarkTools
include("/home/abergman/projects/optimization/SPSA.jl")

w0 = rand(Normal(0,1))
w1 = rand(Normal(0,10))
println("[$w0,$w1]")
sig(x,w0,w1) = logistic( w0 + w1 * x )
x = rand(Uniform(-8,8), 100)
y = map(x -> rand(Bernoulli(sig(x,w0,w1))), x)
plot( x->sig(x,w0,w1) , -7, 7)
scatter!(x, y)

immutable Model
  x::Vector{Float64}
  y::Vector{Float64}
  w::Vector{Float64}
end


# benchmarking different loss functions {{{
function SPSAmod.loss( m::Model )
  sum = 0.0
  len = length(m.x)
  for i = 1:len
    μ = logistic( m.w[1] + m.w[2] * m.x[i] )
    sum += m.y[i]*log(μ) + (1-m.y[i])*log(1-μ)
  end
  return -sum/len
end
@code_warntype SPSAmod.loss(m)
@benchmark SPSAmod.loss(m)

function SPSAmod.loss( m::Model )
  μ = logistic.( m.w[1] + m.w[2] * m.x ) 
  if any(isnan(μ))
    println("m.w: $(m.w)")
    error("μ NaN")
  end
  ϵ = realmin()
  avg = -mean(m.y.*log(ϵ+μ) + (1-m.y).*log(ϵ+(1-μ)))
  if isnan(avg)
    println("m.w: $(m.w)")
    error("avg NaN")
  end
  return avg
end
@code_warntype SPSAmod.loss(m)
@benchmark SPSAmod.loss(m)

function mle{T}(x::Vector{T},y::Vector{T},w::Vector{T})
  sum = 0.0
  len = length(x)
  for i = 1:len
    @inbounds μ = logistic( w[1] + w[2] * x[i] )
    @inbounds sum += y[i]*log(μ) + (1-y[i])*log(1-μ)
  end
  return -sum/len
end # function mle
function SPSAmod.loss( m::Model )
  -mle(m.x,m.y,m.w)
end # function SPSAmod.loss
@code_warntype mle(m.x,m.y,m.w)
@benchmark SPSAmod.loss( m )
#}}}

# Concrete types are faster {{{
type MyType{T<:AbstractFloat}
  a::T
end

func(m::MyType) = m.a+1

code_llvm(func,(MyType{Float64},))
code_llvm(func,(MyType{AbstractFloat},))
code_llvm(func,(MyType,))
#}}}

function Base.append!(Q::Vector, model::Model)
  append!(Q, model.w)
end # function Base.append!

function SPSAmod.unpack!( m::Model, i::Int, Q::Vector)
  SPSAmod.unpack!( m.w, i, Q)
end

m = Model( x, y, ones(2) )
spsa = SPSAmod.SPSA( m )
SPSAmod.clear_history(spsa, true)
SPSAmod.searchADAM( spsa, 1000, 1)
println(m.w)
println("[$w0,$w1]")
plot(spsa)

plot( x->sig(x,w0,w1) , -7, 7)
scatter!(x, y)
plot!( x->sig(x,m.w[1],m.w[2]), -7, 7)

dat = hcat(spsa.param_history...)
plot(dat[1,:])
plot(dat[2,:])


μ = logistic.( m.w[1] + m.w[2] * m.x ) 
any(isnan(log(1-μ)))
any(isnan(log(μ)))
any(isnan(m.y.*log(μ)))
any(isnan((1-m.y).*log(1-μ)))
any(isnan(m.y.*log(μ) + (1-m.y).*log(1-μ)))
avg = -mean(m.y.*log(μ) + (1-m.y).*log(1-μ))

find( isnan, (1-m.y).*log(1-μ) )

log(1-μ[72])
m.y[72]

nextfloat(0.0)

log(eps(0.0)+1e-1200)
