library(rstan)

# Stan code to C++
cpp <- stanc( file="CoinBias.stan" )
page(cpp, method="print")

# compile C++
dso <- stan_model( stanc_ret=cpp )
str(dso)

#create fictitious data
N <- 20
z <- 14
y <- c( rep(1,z), rep(0,N-z) )
dataList <- list(N=N,
                 y=y
                 )

fit <- sampling(dso,
            data=dataList,
            chains=3,
            iter=1000,
            warmup=200,
            thin=1
            )

str(fit, max.level=2) 

str(fit@'sim', max.level=1) 

page( fit, method="print" )

plot( fit )

traceplot( fit )

s <- extract( fit, pars=c("theta"), permuted=TRUE )
names(s)
str(s)

str(s$theta)

hist(s$theta
     , freq=FALSE
     , xlim=c(0,1)
     , breaks=200
     )




