functions {
  real[] LEGI_Amp( real t,
             real[] y,
             real[] theta,
             real[] x_r,
             int[] x_i ) {
    real dydt[3];
    real ta;
    real ti;
    real tr;
    real a;
    real i;
    real r;
    real rTot;
    real kl;
    real kd;
    real kc;
    real L;
    ta   <- theta[1];
    ti   <- theta[2];
    tr   <- theta[3];
    tm   <- theta[4];
    kl   <- theta[5];
    kd   <- theta[6];
    kc   <- theta[6];
    rTot <- theta[7];
    L    <- theta[8];
    a    <- y[1];
    i    <- y[2];
    r    <- y[3];
    dydt[1] <- ta * ( kl * L - kd * a + kc );
    dydt[2] <- ti * ( kl * L - kd * a + kc );
    dydt[3] <- tr * ( a * (rTot - r) - i * r );
    return dydt;
  }
}
data {
  int<lower=1> T;// 
  real y0[3];    // initial state
  real t0;       // initial time
  real ts[T];    // observation times
  real theta[9]; // System parameters
}
transformed data {
  real x_r[0];
  int x_i[0];
}
model {

}
generated quantities {
  real y_hat[T, 3];
  y_hat <- integrate_ode( LEGI_Amp, y0, t0, ts, theta, x_r, x_i );
}


