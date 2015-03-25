
functions { # {{{

} # }}}

data { #{{{
  real<lower=0> r_e;
  real<lower=0> r_l;
  int <lower=1> T;
  int <lower=0> D[T];
} #}}}

transformed data { #{{{
  real log_unif;
  log_unif <- -log(T);
} #}}}

parameters { #{{{
  real<lower=0> e;
  real<lower=0> l;
} #}}}

transformed parameters { #{{{
  vector[T] lp;
  lp <- rep_vector( log_unif, T );
  for (s in 1:T) {
    for (t in 1:s-1) {
      lp[s] <- lp[s] + poisson_log( D[t], e);
    }
    for (t in s:T) {
      lp[s] <- lp[s] + poisson_log( D[t], l);
    }
  }
} #}}}

model { #{{{
  e ~ exponential( r_e );
  l ~ exponential( r_l );

  increment_log_prob( log_sum_exp(lp) );
} #}}}

generated quantities { #{{{

} #}}}
