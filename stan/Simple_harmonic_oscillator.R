library(rstan)# {{{
alarm <- function(  ) {
  system( "ffplay -autoexit ~/Downloads/beep.wav", 
         ignore.stdout=TRUE,
         ignore.stderr=TRUE
         )
}
stancompile <- function( file ) {
  # DESCRIPTION
  # 
  # Args:
  #   x: ARG DESCRIPTION
  # 
  # Returns:
  #   RETURN DESCRIPTION
  cpp <- stanc( file=file )
  dso <- stan_model( stanc_ret=cpp )
  alarm()
  dso
}
# }}}

# sho.stan# {{{
cpp <- stanc( file="./sho.stan" )
dso <- stan_model( stanc_ret=cpp )

T <- 10
dataList <- list( T=T
                 , y0=c(1,1)
                 , t0=0
                 , ts=1:T
                 , theta=c(1,1)
                 )
str(dataList)
fit <- sampling(dso,
            data=dataList,
            chains=1,
            iter=1,
            warmup=0,
            algorithm="Fixed_param"
            )

y_hat <- extract( fit )$y_hat

str( y_hat )

plot(y_hat[,,1])
# }}}

#
# Simulation # {{{
#

y0 <- c(1,1)
ts <- seq( 0, T, length.out=10 )[-1]
T <- length(ts)
noise <- 0.1
dataList <- list( T=T
                 , y0=c(1,1)
                 , t0=0
                 , ts=ts
                 , theta=c(0.15,0)
                 , noise=noise
                 )
str(dataList)

dso <- stancompile( file="./Simple_harmonic_oscillator-Simulator.stan" )

fit.sim <- sampling(dso,
            data=dataList,
            chains=1,
            iter=1,
            warmup=0,
            algorithm="Fixed_param"
            )

y_hat <- extract( fit.sim )$y_hat
str( y_hat )

plot(y_hat[,,1], y_hat[,,2], type='b')
# }}}

# 
# Parameter Estimation # {{{
#
dim(y_hat)
yh <- y_hat
dim(yh)
dim(yh) <- c(T,2)
dim(yh)
str(yh)
yh[,2]

dataList <- list( T=T
                 , y=yh
                 , y0=y0
                 , t0=0
                 , ts=ts
                 , noise=noise
                 )
str(dataList)

dso <- stancompile( file="./Simple_harmonic_oscillator-Estimator.stan" )

fit <- sampling(dso,
            data=dataList,
            chains=4,
            iter=1e3,
            )
print(fit)

plot(fit)

traceplot(fit)

str(fit, max.level=2)

str(fit@sim, max.level=2)

samps <- extract(fit)
str(samps)
hist( samps$theta, breaks=30 )
quantile( samps$theta )

plot( samps$theta, samps$lp__ )

plot( samps$theta, exp(samps$lp__) )

ggplot( s, aes(x=w) ) + 
geom_histogram( aes(y=..density..), binwidth=0.05) + 
geom_density(size=2,color='green') + 
geom_line( data=w.prior, aes(x=x, y=g), color='red' )


# }}}
