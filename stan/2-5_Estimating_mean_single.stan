functions { # {{{

} # }}}

data { #{{{
  real y;
  real priorMu;
  real priorSigma;
} #}}}

transformed data { #{{{

} #}}}

parameters { #{{{
  real mu;
} #}}}

transformed parameters { #{{{

} #}}}

model { #{{{
  mu ~ normal( priorMu, priorSigma );
  y ~ normal(mu,1);
} #}}}

generated quantities { #{{{

} #}}}
