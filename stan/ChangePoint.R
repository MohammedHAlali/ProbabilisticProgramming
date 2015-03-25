library(rstan)
library(ggplot2)
library(dplyr)
library(scales)

# simulated data {{{
r.e <- 0.1
r.l <- 0.1
lambda_e <- 8
lambda_l <- 7
T <- 100
s <- 50

Dt <- c( 
  rpois(   s, lambda_e ),
  rpois( T-s, lambda_l )
)
plot(Dt)

x <- seq(0,20,.1)
plot(x, dexp(x, r.e))
lines(x, dexp(x, r.l))

datalist <- list(
  r_e=r.e,
  r_l=r.l,
  T=T,
  D=Dt
)
# }}}

dso <- stan_model( stanc_ret=stanc(file="./ChangePoint.stan") )

fit <- sampling(
  dso,
  data=datalist,
  chains=4,
  iter=1e3,
)
vars <- c('e','l', 'lp__')
print(fit,vars)

traceplot(fit,vars)

s <- as_data_frame( as.data.frame(fit) )

ggplot(s, aes(x=e)) +
geom_histogram( size=2 )

ggplot(s, aes(x=l)) +
geom_histogram( )

ggplot(s, aes(x=e)) +
geom_density( size=2 )

ggplot(s, aes(x=l)) +
geom_density( size=2 ) +
geom_density( size=2, aes(x=e) )

ggplot(s, aes(x=l, y=e)) + 
geom_point( colour=alpha("black", 1/5) )

ggplot(s, aes(x=l, y=e)) + 
geom_point( colour=lp__ )

lps <- sapply( 1:datalist$T, function(i) paste("lp[",i,"]",sep="") )

ps <- s[lps] %>% summarise_each( funs(mean) ) %>% as.vector() %>% t() %>% exp()
ps <- ps / sum(ps)
plot(ps)
# plot(ps,xlim=c(40,60),lty=2)
sum(ps)

sum((1:T)*ps)


quants <- sapply( lps, 
  function(t) quantile( s[[t]], probs=c(2.5,25,50,75,97.5)/100 ) )
plot(quants["50%",])
lines(quants["50%",],lty=1)
lines(quants["25%",], lty=2)
lines(quants["75%",], lty=2)
lines(quants["2.5%",], lty=3)
lines(quants["97.5%",], lty=3)

