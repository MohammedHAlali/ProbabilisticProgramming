data {
  int<lower=1> N;
  real y[N];
}
parameters {
  real mu;
}
model {
  y ~ normal( mu, 1 )
}
