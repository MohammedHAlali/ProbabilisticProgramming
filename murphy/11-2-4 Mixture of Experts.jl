using Distributions
using StatsFuns

x = 0:0.1:10 |> collect

softmax( [1,a] [1,2,4] )

V = [1 1; 2 1; 3 1]

X = [ones(x) x]

Xi = X[1,:]

[ softmax(V*X[i,:]) for i in 1:length(x) ]

