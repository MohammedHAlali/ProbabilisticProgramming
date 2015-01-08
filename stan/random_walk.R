library(rstan)
library(ggplot2)

dso <- stan_model( stanc_ret=stanc(file="./random_walk.stan") )

# inverse gamma plot # {{{
x <- seq( 0, 5, length.out=100 )
g <- dgamma( x, shape=1, rate=1 )
plot(x, g)
w.prior <- data.frame(
  x=x,
  g=g
)
# }}}

# random walk simulated data # {{{
y <- rnorm(1e2, mean=0, sd=2)
y <- cumsum( y )
plot(y, type="b", ylim=30*c(-1,1))
datalist <- list(
  T=length(y),
  y=y,
  v=0.1
)
# }}}

fit <- sampling(
  dso,
  data=datalist,
  chains=4,
  iter=1e3
)
print(fit)
s <- as.data.frame( fit )

traceplot(fit)

ggplot( s, aes(x=w) ) + 
geom_histogram( aes(y=..density..), binwidth=0.05) + 
geom_density(size=2,color='green') + 
geom_line( data=w.prior, aes(x=x, y=g), color='red' )

s["theta[1]"]

plot( s[["w"]], s[["lp__"]] )

q <- quantile( s[["theta[1]"]] )

thetas <- sapply( 1:datalist$T, function(i) paste("theta_ppc[",i,"]",sep="") )

quants <- sapply( thetas, 
  function(t) quantile( s[[t]], probs=c(2.5,25,50,75,97.5)/100 ) )
str(quants)

plot(y,type="l",ylim=30*c(-1,1))
lines(quants["50%",],lty=1)
lines(quants["25%",], lty=2)
lines(quants["75%",], lty=2)
lines(quants["2.5%",], lty=3)
lines(quants["97.5%",], lty=3)


