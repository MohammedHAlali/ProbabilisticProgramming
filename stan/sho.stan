functions { # {{{
  real[] sho(real t,       // time
             real[] y,     // system state 
             real[] theta, // parameters
             real[] x,     // real data
             int[] x_int   // integer data
             ) {
    real dydt[2];
    dydt[1] <- -y[1];
    dydt[2] <- -y[2];
    return dydt;
  }
} # }}}

data { #{{{
  int<lower=1> T;
  real y0[2];
  real t0;
  real ts[T];
  real theta[2];
} #}}}

transformed data { #{{{
  real x[0];
  int x_int[0];
} #}}}

parameters { #{{{

} #}}}

transformed parameters { #{{{
} #}}}

model { #{{{

} #}}}

generated quantities { #{{{
  real y_hat[T,2];
  y_hat <- integrate_ode(
    sho,   // function
    y0,    // initial state
    t0,    // initial time
    ts,    // solution times
    theta, // parameters
    x,     // real data
    x_int  // integer data
  );
} #}}}
