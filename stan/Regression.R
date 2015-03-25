library(rstan)
library(ggplot2)
library(dplyr)
library(scales)

# simluated data {{{
X <- seq( -5, 5, .01)
x <- runif( 100, min=-5, max=5)
K <- 3
w.true <- rnorm( K )
phi <- function(x) {
	c( 1, x, sin(x) )
}
phi <- Vectorize( phi )
y <- function( x, w ) {
	w.true %*% phi(x)
}
plot( X, y(X, w.true), type="l", lwd=3 )
f <- y(x,w.true) + rnorm( length(x) , sd=1 )
f <- drop(f)
points( x, f)
alpha=0.01

datalist <- list(
  N=length(x),
  K=K,
  x=t(phi(x)),
  y=f,
	alpha=alpha
)	

print(w.true)

# }}}

dso <- stan_model( stanc_ret=stanc(file="./Regression.stan") )

fit <- sampling( 
  dso,
	data=datalist,
	chains=4,
	iter=1e3
)
print(fit)

traceplot(fit)

pairs(fit)

s <- as_data_frame( as.data.frame(fit) )
print(s)
names(s)
str(s)
summary(s)

ggplot(s, aes(x=sigma)) +
geom_histogram( binwidth=0.01 )

w.df <- as_data_frame(as.data.frame(extract( fit, pars=c("w") )$w))
print(w.df)

ggplot(w.df, aes(x=V1)) +
geom_density( color='blue' ) +
geom_histogram( fill='blue', binwidth=0.02, aes(y=..density..)) +
geom_vline( xintercept=w.true[1], color='blue' ) +
geom_density( aes(x=V2), color='green' ) +
geom_vline( xintercept=w.true[2], color='green' ) +
geom_density( aes(x=V3), color='red' ) +
geom_vline( xintercept=w.true[3], color='red' )

W <- extract(fit, pars=c("w"))$w
str(W)
W.samp <- W[sample(nrow(W), 80),]
str(W.samp)

PHI <- t(phi(X))
str(PHI)

Y <- W.samp %*% t(PHI)
Y <- t(Y)
dim(Y) <- NULL
length(Y)


dataset <- data.frame( x=x, y=f)
df <- data.frame( x=rep( X, nrow(W.samp)) , y=Y )
# print(df)
nrow(df)


ggplot( df, aes( x=x, y=y)) +
geom_point( colour=alpha("black",5e-3), size=3 ) + 
geom_point( data=dataset, aes( x=x, y=f), color='black', size=5 ) +
geom_point( data=dataset, aes( x=x, y=f), color='green', size=3 )

ggplot( df, aes( x=x, y=y)) +
stat_density2d(geom="tile", aes(fill=..density..), contour=FALSE ) + 
geom_point( data=dataset, aes( x=x, y=f), color='black', size=5 ) +
geom_point( data=dataset, aes( x=x, y=f), color='green', size=3 )


PHI <- t(phi(x))
S0 <- 1/alpha*diag( K )
str(S0)

iSN <- solve(S0) + t(PHI) %*% PHI 
str(iSN)
SN <- solve(SN)

mN <- SN %*% ( solve(S0) %*% m0 + beta * t(PHI) %*% f )


w.ML <- solve( t(PHI) %*% PHI, t(PHI) %*% f ) %>% drop()

w.bayesian <- apply( W, 2, mean)
str(w.bayesian)

print( cbind(w.ML, w.bayesian) )

k <- function(x) {
	t(phi(x)) %*% SN %*% phi(1) %>% drop()
}

curve( k, -5, 5)
