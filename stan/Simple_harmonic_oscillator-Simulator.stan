/* Simulator
 * Noisy measurements from Simple Harmonic Oscillator
 * p. 163
 */
functions { #{{{
  real[] sho(real t,
             real[] y, 
             real[] theta,
             real[] x_r,
             int[] x_i) {
    real dydt[2];
    dydt[1] <- y[2];
    dydt[2] <- -y[1] - theta[1] * y[2];
    return dydt;
  }
} # }}}
data {/*{{{*/
  int<lower=1> T;// 
  real y0[2];    // initial state
  real t0;       // initial time
  real ts[T];    // observation times
  real theta[2]; // System parameter
  real noise;    // Observation noise
}/*}}}*/
transformed data {/*{{{*/
  real x_r[0];
  int x_i[0];
}/*}}}*/
parameters {/*{{{*/
  
}
model {

}/*}}}*/
generated quantities { # {{{
  real y_hat[T,2];
  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x_r, x_i );

  // add measurement error
  for (t in 1:T) {
    y_hat[t,1] <- y_hat[t,1] + normal_rng(0,noise);
    y_hat[t,2] <- y_hat[t,2] + normal_rng(0,noise);
  }
} # }}}
