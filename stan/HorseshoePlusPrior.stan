// http://arxiv.org/pdf/1502.00560v2.pdf
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
  vector<lower=0>[N] eta;
  real<lower=0> tau;
} #}}}

transformed parameters { #{{{
  vector[N] w;
  w <- tau * (eta .* lambda .* w_raw);
} #}}}

model { #{{{
  w_raw ~ normal(0,1); // implies w ~ normal(0,tau)
  lambda ~ cauchy( 0, 1 ); // local shrinkage
  eta ~ cauchy(0,1); // local shrinkage
  tau ~ cauchy(0,1); // global shrinkage
  // tau <- p_n / n; // empirical Bayes estimator
} #}}}

generated quantities { #{{{

} #}}}
