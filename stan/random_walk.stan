functions { # {{{

} # }}}
data { #{{{
  int<lower=1> T; // number of time points
  real y[T];      // initial state
  real v;         // observation variance
} #}}}
transformed data { #{{{

} #}}}
parameters { #{{{
  real<lower=0> w;
  real theta[T];
} #}}}
transformed parameters { #{{{

} #}}}
model { #{{{
  w ~ gamma( 1, 1 );

  theta[1] ~ normal( y[1], w );
  for (t in 2:T) {
    theta[t] ~ normal( theta[t-1], w );
  }  
  for (t in 1:T) {
    y[t] ~ normal( theta[t], v );
  }
} #}}}
generated quantities { #{{{
  real y_ppc[T];
  real theta_ppc[T];

  theta_ppc[1] <- normal_rng( y[1], w );
  for (t in 2:T) {
    theta_ppc[t] <- normal_rng( theta_ppc[t-1], w );
  }  
  for (t in 1:T) {
    y_ppc[t] <- normal_rng( theta_ppc[t], v );
  }
} #}}}
