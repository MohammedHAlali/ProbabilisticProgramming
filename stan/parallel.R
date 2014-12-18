library(rstan)
library(parallel)

#create fictitious data
N <- 20
z <- 14
y <- c( rep(1,z), rep(0,N-z) )
dataList <- list(N=N,
                 y=y
                 )

f1 <- stan( file="CoinBias.stan"
           , data=dataList
           , chains=1
           , iter=1
           )

seed <- 12345

sflist1 <- mclapply(1:8
                    , mc.cores=8
                    , function(i) {
                      stan( fit=f1
                           , seed=seed
                           , data=dataList
                           , iter=1000000
                           , chains=1
                           , chain_id=i
                           , refresh=-1
                           )
                    } 
                    )

fit <- sflist2stanfit( sflist1 )
fit

s <- extract( fit, pars=c("theta") )
str(s)

hist(s$theta
     , freq=FALSE
     , xlim=c(0,1)
     , breaks=200
     )




