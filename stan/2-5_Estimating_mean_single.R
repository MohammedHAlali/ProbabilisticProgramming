library(rstan)
library(ggplot2)

cpp <- stanc( file="./2-5_Estimating_mean_single.stan" )
dso <- stan_model( stanc_ret=cpp )

dataList <- list( y=3,
                  priorMu=0,
                  priorSigma=1)

fit <- sampling(dso,
            data=dataList,
            chains=4,
            iter=1e4,
            warmup=0
            )
print(fit)
traceplot(fit)

samps <- as.data.frame(extract(fit))
str(samps)

ggplot( samps, aes(x=mu) ) + geom_density()

ndist <- data.frame(x=seq(-3, 6, length.out=100 ))
ndist$n <- dnorm( ndist$x, dataList$priorMu, dataList$priorSigma)
ggplot(ndist, aes(x=x,y=n)) + geom_line() 

ggplot( samps, aes(x=mu) ) + 
geom_histogram( aes(y=..density..), binwidth=0.1) + 
geom_density(size=2,color='green') + 
geom_line( data=ndist, aes(x=x, y=n), color='red' )

hist(samps$mu)

x <- seq(0, 3, .01)
ig <- 1/dgamma( x, 0.001, scale=0.1 )
plot(x,ig,type='l')
