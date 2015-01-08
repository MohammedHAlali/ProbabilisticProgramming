/* Parameter Estimator
 * Noisy measurements from Simple Harmonic Oscillator
 * p. 163
 */
functions {
  real[] sho(real t,
             real[] y, 
             real[] theta,
             real[] x,
             int[] x_int) {
    real dydt[2];
    dydt[1] <- y[2];
    dydt[2] <- -y[1] - theta[1] * y[2];
    return dydt;
  }
}
data {
  int<lower=1> T;// 
  real y[T,2];   // initial state
  real y0[2];    // initial state
  real t0;       // initial time
  real ts[T];    // observation times
  real noise;    // observation noise
}
transformed data {
  real x[0];
  int x_int[0];
}
parameters {
  real<lower=0> theta[1];
}
model {
  real y_hat[T,2];
  theta ~ normal(0,1);
  /* print( "theta[1] = ", theta[1] ) */
  y_hat <- integrate_ode(sho, y0, t0, ts, theta, x, x_int);
  for (t in 1:T) {
    /* y_hat[t,1] <- y[t,1] - theta[1]; */
    /* y_hat[t,2] <- y[t,2] - theta[1]; */
    y[t] ~ normal(y_hat[t], noise);
    /* print( "y[", t, ",1] = ", y[t,1] ) */
    /* print( "y[", t, ",2] = ", y[t,2] ) */
    /* print( "y_hat[", t, ",1] = ", y_hat[t,1] ) */
    /* print( "y_hat[", t, ",2] = ", y_hat[t,2] ) */
  }
}
