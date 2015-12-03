functions { # {{{

} # }}}

data { #{{{
  int<lower=1> N;
} #}}}

transformed data { #{{{

} #}}}

parameters { #{{{
  vector[N] w_raw;
  vector<lower=0>[N] lambda;
  real<lower=0> tau;
} #}}}

transformed parameters { #{{{
  vector[N] w;
  w <- tau * (lambda .* w_raw);
} #}}}

model { #{{{
  w_raw ~ normal(0,1); // implies w ~ normal(0,tau)
  lambda ~ cauchy( 0, 1 ); // local shrinkage
  tau ~ cauchy(0,1); // global shrinkage
} #}}}

generated quantities { #{{{

} #}}}
