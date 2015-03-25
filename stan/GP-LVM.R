library(rstan)
library(ggplot2)
library(dplyr)

# simulated data {{{
N <- 80
x <- runif(N, -3,3)
y <- sapply( x, function(x) c(x, x^2)+rnorm(2, 0, 0.4) )
plot( y[1,], y[2,] )
datalist <- list(
	D = 2,
	d = 1,
	N = N,
	y = y
)

plot(x, dcauchy( x, location=0, scale=1))

# }}}


dso <- stan_model( stanc_ret=stanc(file="./GP-LVM.stan") )

op <- optimizing( dso, data=datalist )

str(op)

op$par

plot(op$par)



# PCA {{{
y.pca <- prcomp( t(y), center=FALSE, scale=FALSE )
y.proj <- predict( y.pca, newdata=t(y))
plot(y.proj)

print(y.pca)
str(y.pca)
summary(y.pca)

# }}}

kernel <- function( x1, x2 ) {
	a1 * exp( -a2 * (x1 - x2)^2 )
}
kernel <- Vectorize(kernel)

init <- list(
  # x = matrix(.01*rnorm(N), N, 1),
  # x = matrix(seq(0,10, length.out=N), N, 1),
  x = matrix(y.proj[,2], N, 1), 
	noise = 1,
	a1 = 1,
	a2 = 1
)
op <- optimizing( dso, data=datalist, init=init )
print( op$par['a1'] )
print( op$par['a2'] )
print( op$par['noise'] )
a1 <- op$par['a1']
a2 <- op$par['a2']
X.fit <- sapply( seq(N), function(i) op$par[paste0("x[",i,",1]")] )
str(X.fit)
X.star <- seq(min(X.fit), max(X.fit), length.out=10*N)
K <- outer( X.fit, X.fit, kernel)
U <- chol( K + op$par['noise'] * diag(N) )
alpha <- backsolve( U, backsolve( U, t(y), transpose=TRUE) ) 
str(alpha)
K.star <- sapply( X.star, function(x) kernel( x, X.fit ) )
f.star <- t(K.star) %*% alpha
plot( f.star[,1], f.star[,2], type='o' )
points( y[1,], y[2,], col='green', lwd=4 )

plot(X.star, f.star[,1] )
points(X.fit, y[1,], col='green',pch=16)

plot(X.star, f.star[,2] )
points(X.fit, y[2,], col='green', pch=16)

# mcmc sampling {{{
init <- list(
  x = matrix(y.proj[,2], N, 1), 
	noise = 0.2,
	a1 = 1,
	a2 = 0.1
)

fit <- sampling( dso, data=datalist, iter=10000, chains=1,
								init=list(init))

fit@inits

str(fit)

print(fit)

traceplot(fit, pars="x")

traceplot(fit, pars=c("a1","a2", "noise", "lp__"))

df <- as_data_frame(as.data.frame(fit))
df <- df %>% arrange(  desc(`lp__`)  ) 
names(df) 

plot( as.vector(df[,"lp__"]) )

hist( df[,"x[1,1]"] )

ggplot( df, aes(`lp__`, `noise` ) ) +
  geom_point( aes( size=1 ) )

for ( idx in seq(1,50,length.out=50) ) {
	X.fit <- sapply( seq(N), function(i) df[idx,paste0("x[",i,",1]")] )
	X.fit <- unlist(X.fit)
	X.star <- seq(min(X.fit), max(X.fit), length.out=10*N)
	K <- outer( X.fit, X.fit, kernel)
	U <- chol( K + unlist(df[idx,"noise"]) * diag(N) )
	alpha <- backsolve( U, backsolve( U, t(y), transpose=TRUE) )
	K.star <- sapply( X.star, function(x) kernel( x, X.fit ) )
	f.star <- t(K.star) %*% alpha
	plot( f.star[,1], f.star[,2], type='o', xlim=c(-4,4), ylim=c(-1,10) )
	points( y[1,], y[2,], col='green', lwd=4 )
	# plot(X.star, f.star[,1], xlim=c(-10, 10), ylim=c(-4,4) )
	# points(X.fit, y[1,], col='green', pch=16)
}


# }}}
