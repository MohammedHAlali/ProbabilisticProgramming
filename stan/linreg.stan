functions { # {{{

} # }}}

data { #{{{
  int<lower=1> N; 
  vector[N] x;
  vector[N] y;
} #}}}

transformed data { #{{{

} #}}}

parameters { #{{{
  real alpha;
  real beta;
  real<lower=0> sigma;
} #}}}

transformed parameters { #{{{
  vector[N] r;
  r <- y - (alpha - beta * x);
} #}}}

model { #{{{
  r ~ normal(0,1);
} #}}}

generated quantities { #{{{
  vector[N] y_ppc;
  for (i in 1:N) {
    y_ppc[i] <- normal_rng( alpha+beta*x[n], sigma );
  }

} #}}}
