functions { # {{{
  matrix GenCovYY( vector x, real rho_sq, real eta_sq, real sigma_sq ) {
    int N;
    N <- num_elements( x );
    {
      matrix[N,N] Sigma;
      for (i in 1:N)
        for (j in 1:N)
          Sigma[i,j] <- exp( -0.5 * rho_sq * pow( x[i] - x[j], 2)) * eta_sq;
      for (i in 1:N) {
        Sigma[i,i] <- Sigma[i,i] + sigma_sq;
      }
      return Sigma;
    }
  }
  matrix GenCovWY( vector w, vector x, real rho_sq, real eta_sq ) {
    int Nw;
    int Nx;
    Nw <- num_elements( w );
    Nx <- num_elements( x );
    {
      matrix[Nw,Nx] Sigma;
      for (m in 1:Nw)
        for (n in 1:Nx)
          Sigma[m,n] <- exp( -0.5 * rho_sq * pow( w[m] - x[n], 2)) * eta_sq * (w[m] - x[n]) * -rho_sq;
      return Sigma;
    }
  }
  matrix GenCovW( vector w, real rho_sq, real eta_sq, real sigma_sq ) {
    int N;
    N <- num_elements( w );
    {
      matrix[N,N] Sigma;
      for (i in 1:N)
        for (j in 1:N)
          Sigma[i,j] <- exp( -0.5 * rho_sq * pow(w[i] - w[j], 2)) * eta_sq * (1 - rho_sq*pow(w[i] - w[j],2)) * rho_sq;
      for (i in 1:N) {
        Sigma[i,i] <- Sigma[i,i] + sigma_sq;
      }
      return Sigma;
    }
  }
  matrix GenCovWW( vector w1, vector w2, real rho_sq, real eta_sq ) {
    int N1;
    int N2;
    N1 <- num_elements( w1 );
    N2 <- num_elements( w2 );
    {
      matrix[N1,N2] Sigma;
      real norm;
      for (i in 1:N1)
        for (j in 1:N2) {
          norm <- pow(w1[i] - w2[j], 2);
          Sigma[i,j] <- exp( -0.5 * rho_sq * norm) * eta_sq * (1 - rho_sq * norm) * rho_sq;
        }
      return Sigma;
    }
  }
} # }}}

data { #{{{
  int<lower=1> N1;
  vector[N1] x1;
  vector[N1] y1;
  int<lower=1> N2;
  vector[N2] x2;
  
  real<lower=0> rho_sq;
  real<lower=0> eta_sq;
  real<lower=0> sigma_sq_y;
  real<lower=0> sigma_sq_w;
} #}}}

transformed data { #{{{
  vector[2*N2] mu;
  matrix[2*N2,2*N2] L;

  {
    matrix[N1,N1] Sigma;   // cov(w,w)
    matrix[N2,N2] Omega11; // kyy(x1,x1)
    matrix[N2,N2] Omega22; // kww(x2,x2)
    matrix[N2,N2] Omega12; // kwy(x2,x2)
    matrix[N1,2*N2] K;     // [ kwy(x1,x2) , kww(x1,x2) ]
    matrix[2*N2,N1] K_transpose_div_Sigma;
    matrix[2*N2,2*N2] Tau;
    matrix[2*N2,2*N2] Omega; // Omega = [[Omega11, Omega12], [Omega12', Omega22]]

    Sigma   <- GenCovW(  x1, rho_sq, eta_sq, sigma_sq_w );
    Omega11 <- GenCovYY( x2, rho_sq, eta_sq, sigma_sq_y );
    Omega22 <- GenCovW(  x2, rho_sq, eta_sq, sigma_sq_w );
    Omega12 <- GenCovWY( x2, x2, rho_sq, eta_sq);
    Omega <- append_row( append_col( Omega11 , Omega12 )
                       , append_col( Omega12', Omega22 ));


    K <- append_col( GenCovWY( x1, x2, rho_sq, eta_sq)
                   , GenCovWW( x1, x2, rho_sq, eta_sq) );

    K_transpose_div_Sigma <- K' / Sigma;
    mu <- K_transpose_div_Sigma * y1;
    Tau <- Omega - K_transpose_div_Sigma * K;
    for (i in 1:N2)
      for (j in (i+1):N2)
        Tau[i,j] <- Tau[j,i];

    L <- cholesky_decompose(Tau);
  }
} #}}}

parameters { #{{{
  vector[2*N2] z;
} #}}}

transformed parameters { #{{{

} #}}}

model { #{{{
  z ~ normal(0,1);
} #}}}

generated quantities { #{{{
  vector[N2] y;
  vector[N2] w;
  vector[2*N2] y2;    // y2 ~ multi_normal(mu, Tau);
  y2 <- mu + L * z;
  y <- head(y2,N2);
  w <- tail(y2,N2);
} #}}}
