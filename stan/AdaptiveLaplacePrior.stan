functions { # {{{

} # }}}

data { #{{{
  int<lower=1> N;
  real<lower=0> alpha;
  real<lower=0> beta;
} #}}}

transformed data { #{{{

} #}}}

parameters { #{{{
  vector[N] w_raw;
  vector<lower=0>[N] g;
  vector<lower=0>[N] tau;
} #}}}

transformed parameters { #{{{
  vector[N] w;
  w <- tau .* w_raw;
} #}}}

model { #{{{
  w_raw ~ normal(0,1); // implies w ~ normal(0,tau)
  tau ~ gamma( 1, (g .* g)/2. );
  g ~ inv_gamma(alpha,beta);
} #}}}

generated quantities { #{{{
  real g_sim;
  g_sim <- inv_gamma_rng(alpha,beta);
} #}}}
