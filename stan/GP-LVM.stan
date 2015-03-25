data {
	int<lower=1> N;
	int<lower=1> D;
	int<lower=1> d;
	/* vector[N] y[D]; */
	matrix[D,N] y;
}
transformed data {
	vector[D] w;
	vector[N] mu;
	for( i in 1:N) {
		mu[i] <- 0;
	}

	for( i in 1:D) {
		w[i] <- 1;
	}
}
parameters {
	vector[d] x[N];
	real<lower=0> noise;
	real<lower=0> a1;
  real<lower=0> a2;
}
model {
	matrix[N,N] Kx;
	matrix[N,N] L;
	for (j in 1:N) {
		for (i in j:N) {
			Kx[i,j] <- a1*exp( -a2*dot_self( x[i] - x[j] ) );
			Kx[j,i] <- Kx[i,j];
		}
	}

	for (k in 1:N) {
		Kx[k,k] <- Kx[k,k] + noise;
	}

	L <- cholesky_decompose( Kx );

	/* for( p in 1:D ) { */
	/* 	y[p] ~ multi_normal_cholesky( mu, L ); */
	/* } */

	for (i in 1:N) {
		x[i] ~ normal( 0, 4);
	}
	y ~ multi_gp_cholesky( L, w);
	noise ~ cauchy(0,1);
	a1 ~ cauchy(0,1);
	a2 ~ cauchy(0,1);
}
