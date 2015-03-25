/* p.129 
*/
functions { # {{{

} # }}}

data { #{{{
  int<lower=1> N;
  real x[N];
  real<lower=0> s;
  real<lower=0> rho_sq;
  real<lower=0> eta_sq;
} #}}}

transformed data { #{{{
  vector[N] mu;
  cov_matrix[N] Sigma;
  mu <- rep_vector( 0, N );
  // matrix: column-major
  // for c in col
  //   for r in row
  for (j in 1:N) {
    for (i in 1:N) {
      Sigma[i,j] <- eta_sq * exp( -rho_sq * pow(x[i]-x[j],2) ) + 
        if_else( i==j, s, 0.0 );
    }
  }
} #}}}

parameters { #{{{
  vector[N] y;
} #}}}

transformed parameters { #{{{

} #}}}

model { #{{{
  y ~ multi_normal( mu, Sigma );
} #}}}

generated quantities { #{{{

} #}}}
