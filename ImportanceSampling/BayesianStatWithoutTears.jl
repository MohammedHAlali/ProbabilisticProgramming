module Example5 #{{{

using Distributions

t1 = 0.5
t2 = 0.3

N = 20000

instrumentalDist = rand(N,2)

function sample(t1, t2)
  x1 = Binomial(5, t1)
  x2 = Binomial(6, t1)
  x3 = Binomial(4, t1)

  z1 = Binomial(5, t2)
  z2 = Binomial(4, t2)
  z3 = Binomial(6, t2)

  y = zeros(3)
  y[1] = rand(x1) + rand(z1)
  y[2] = rand(x2) + rand(z2)
  y[3] = rand(x3) + rand(z3)

  return y
end

function likelihood(t1,t2)
    
  x1 = Binomial(5, t1)
  x2 = Binomial(6, t1)
  x3 = Binomial(4, t1)

  z1 = Binomial(5, t2)
  z2 = Binomial(4, t2)
  z3 = Binomial(6, t2)

  px1 = pdf(x1)
  px2 = pdf(x2)
  px3 = pdf(x3)

  pz1 = pdf(z1)
  pz2 = pdf(z2)
  pz3 = pdf(z3)

  py1 =
    px1[2+1] * pz1[5+1] +
    px1[3+1] * pz1[4+1] +
    px1[5+1] * pz1[2+1] +
    px1[4+1] * pz1[3+1]

  py2 = 
    px2[1+1] * pz2[4+1] +
    px2[2+1] * pz2[2+1] +
    px2[4+1] * pz2[1+1] +
    px2[5+1] * pz2[0+1]

  py3 = 
    px3[0+1] * pz3[6+1] +
    px3[1+1] * pz3[5+1] +
    px3[2+1] * pz3[4+1] +
    px3[3+1] * pz3[3+1] +
    px3[4+1] * pz3[2+1]

  lp = log(py1) + log(py2) + log(py3)

  joint = py1 * py2 * py3

  return joint

end

l = zeros(N)
for i = 1:N
  t1 = instrumentalDist[i,1]
  t2 = instrumentalDist[i,2]
  l[i] = likelihood( t1, t2)
end
q = l / sum(l)

resampleDist = Categorical(q)

resampleIdx = rand(resampleDist,N)

resampleVals = instrumentalDist[resampleIdx,1:2]

end # Example5 }}}

