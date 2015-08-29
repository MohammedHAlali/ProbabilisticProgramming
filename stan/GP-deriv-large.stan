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
    real norm;
    int N;
    N <- num_elements( w );
    {
      matrix[N,N] Sigma;
      for (i in 1:N)
        for (j in 1:N) {
          norm <- pow(w[i] - w[j], 2);
          Sigma[i,j] <- exp( -0.5 * rho_sq * norm) * eta_sq * (1 - rho_sq * norm) * rho_sq;
        }
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
  vector[N1] w1;
  int<lower=1> N2;
  vector[N2] x2;
  
  real<lower=0> rho_sq;
  real<lower=0> eta_sq;
  real<lower=0> sigma_sq_y;
  real<lower=0> sigma_sq_w;
} #}}}

transformed data { #{{{
  vector[N1+2*N2] mu;
  matrix[N1+2*N2,N1+2*N2] L;
  int NN;

  NN <- N1+2*N2;

  mu <- rep_vector( 0, NN);

  {
    matrix[N1,N1] Kww;     // kw(x1,x1)
    matrix[N1,N2] Ky;      // kwy(x1,x2)
    matrix[N1,N2] Kw;      // kww(x1,x2)
    matrix[N2,N2] Omega11; // kyy(x1,x1)
    matrix[N2,N2] Omega22; // kww(x2,x2)
    matrix[N2,N2] Omega12; // kwy(x2,x2)
    matrix[NN,NN] Sigma;

    Kww     <- GenCovW(  x1, rho_sq, eta_sq, sigma_sq_w );
    Omega11 <- GenCovYY( x2, rho_sq, eta_sq, sigma_sq_y );
    Omega22 <- GenCovW(  x2, rho_sq, eta_sq, sigma_sq_w );
    Omega12 <- GenCovWY( x2, x2, rho_sq, eta_sq);
    Ky      <- GenCovWY( x1, x2, rho_sq, eta_sq);
    Kw      <- GenCovWW( x1, x2, rho_sq, eta_sq);

    Sigma <- append_row( append_row( 
        append_col( append_col( Kww,      Ky ),      Kw )
      , append_col( append_col( Ky', Omega11 ),-Omega12 ))
      , append_col( append_col( Kw',-Omega12'), Omega22 ));

    L <- cholesky_decompose(Sigma);
    
    // print( "Kww = ", Kww );
    // print( "Ky = ", Ky );
    // print( "Kw = ", Kw );
    // print( "Omega11 = ", Omega11 );
    // print( "Omega22 = ", Omega22 );
    // print( "Omega12 = ", Omega12 );
    // print( "Sigma = ", Sigma );
    // print( "L = ", L );

  }
} #}}}

parameters { #{{{
  vector[2*N2] y2w2; // [y, w] at x2
} #}}}

transformed parameters { #{{{
} #}}}

model { #{{{
  vector[NN] y; // y = [w1, y2, w2]
  for (n in 1:N1) {
    y[n] <- w1[n];
  }
  for (n in 1:2*N2) {
    y[n+N1] <- y2w2[n];
  }
  y ~ multi_normal_cholesky( mu, L );
} #}}}

generated quantities { #{{{
  vector[N2] y2;
  vector[N2] w2;

  y2 <- head(y2w2,N2);
  w2 <- tail(y2w2,N2);
} #}}}
