module Ex38CauchyTailProb #{{{

using Distributions

dist = Cauchy(0,1)

M = 10000

x = rand(dist, M)

function heaviside(x)
  0.5 * (sign(x) + 1)
end

h1 = heaviside(x-2)

var(h1)

p1 = sum(h1) / M

# p2
h2 = 0.5*heaviside(abs(x) - 2)

var(h2)

p2 = sum(h2) / M

# p3
u = 2*rand(M)

h3 = 0.5 - 2 ./ (pi*(1+u.^2))

p3 = sum(h3)/M

var(h3)

# p4
u = 0.5*rand(M)

h4 = u.^-2 ./ (2*pi*(1+u.^-2))

p4 = sum(h4) / M

using DataFrames

df = DataFrame(
  h1=h1,
  h2=h2,
  h3=0.5-h3,
  h4=h4
)

df_avg = DataFrame(
  h=["h1","h2","h3","h4"],
  mean=[p1,p2,p3,p4],
  var=map(var,(h1,h2,h3,h4)) |> collect
)

function onlineMeanVar(x::Array{Float64,1})
  M = length(x)
  mean = zeros(x)
  var = zeros(x)
  mean[1] = x[1]
  M2 = 0
  for m = 2:M
    delta = x[m] - mean[m-1]
    mean[m] = mean[m-1] + delta / m
    M2 += delta * (x[m] - mean[m])
    var[m] = M2 / m
  end
  return (mean, var)
end

df_iter = DataFrame(m=1:M)
(df_iter[:m1],df_iter[:v1]) = onlineMeanVar(h1)
(df_iter[:m2],df_iter[:v2]) = onlineMeanVar(h2)
(df_iter[:m3],df_iter[:v3]) = onlineMeanVar(h3)
(df_iter[:m4],df_iter[:v4]) = onlineMeanVar(h4)

df_iter[:se1] = sqrt(df_iter[:v1]) ./ sqrt(collect(1:M))
df_iter[:se2] = sqrt(df_iter[:v2]) ./ sqrt(collect(1:M))
df_iter[:se3] = sqrt(df_iter[:v3]) ./ sqrt(collect(1:M))
df_iter[:se4] = sqrt(df_iter[:v4]) ./ sqrt(collect(1:M))

end # Ex38CauchyTailProb }}}

