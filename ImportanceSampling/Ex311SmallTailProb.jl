module Ex311SmallTailProb #{{{

include("Common.jl")

using Distributions

M = 100000
a = 4.5
X = rand(Exponential(1), M)
Y = a + X

(F, err) = quadgk( x -> e^-x, a, Inf)

phi = x -> pdf(Normal(0,1), x)
fY = x -> e.^-x ./ F

w = phi(Y) ./ fY(Y)

mean = sum(w) / M

sol = 1-cdf(Normal(0,1),4.5)

(mean - sol) / sol

var(w)

(mean_m, var_m) = Common.onlineMeanVar(w)

end # Ex311SmallTailProb }}}

