functions { # {{{

} # }}}

data { #{{{
  int N;
  matrix[N,N] W;
  vector[N] a;
} #}}}

transformed data { #{{{
  matrix[N,N] L;
  vector[N] zero;
  L <- cholesky_decompose(W);
  zero <- rep_vector( 0, N);
} #}}}

parameters { #{{{
  vector[N] x;
} #}}}

transformed parameters { #{{{

} #}}}

model { #{{{
  x ~ multi_normal_cholesky( zero, L );
  for (i in 1:N) {
    increment_log_prob( log( 1 + exp( x[i] + a[i] )) );
  }

} #}}}

generated quantities { #{{{

} #}}}
