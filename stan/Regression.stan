data {
	int<lower=0> N; // Number of data items
	int<lower=0> K; // Number of predictors
	matrix[N,K]  x; // design matrix
  vector[N]    y; // observed output
	real<lower=0> alpha; // prior scale
}
parameters {
	vector[K] w;				 // weight parameters
	real<lower=0> sigma; // noise scale
}
model {
	w ~ normal( 0, 1/alpha);    // prior
	y ~ normal( x * w, sigma ); // likelihood
}


