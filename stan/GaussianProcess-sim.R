library("rstan")
library("ggplot2")

s <- 1e-1
x <- (-50:50) / 10
N <- length(x)
rho_sq <- 1e0 
eta_sq <- 1e0
plot( x, eta_sq*exp(-rho_sq*x^2) )
lines( x, rep(s, length(x)) )

fit_sim <- 
  stan(
       file="./GaussianProcess-sim.stan",
       fit=fit_sim,
       data=list(
                 x=x,
                 s=s,
                 N=N,
                 rho_sq=rho_sq,
                 eta_sq=eta_sq
                 ),
       iter=200,
       chains=3
       )
ss <- extract( fit_sim, permuted=TRUE )

traceplot( fit_sim )

str( ss )

df <- data.frame( x=x, y_sim=colMeans(ss$y) )
plot( x, df$y_sim )
plot( x, ss$y[1,] )
lines( x, ss$y[2,] )
lines( x, ss$y[3,] )
lines( x, ss$y[4,] )
lines( x, ss$y[5,] )

plot <- qplot( x, y_sim, data=df, xlim=c(-5,5), ylim=c(-4,4) )

str(ss$y[1,])

str(colMeans(ss$y)) 
