library(rstan)

schools_dat <- list(J=8,
                    y=c(28, 8, -3, 7, -1, 1, 18, 12),
                    sigma=c(15, 10, 16, 11, 9, 11, 10, 18))

fit1 <- stan(file='Eight_Schools.stan',
            data=schools_dat,
            iter=1000,
            chains=4
            )

# fit the model again without recompiling the model
fit2 <- stan(fit=fit1,
             data=schools_dat,
             iter=10000,
             chains=4
             )

print( fit1 )
plot(fit1)

# return list of arrays
la <- extract( fit2, permuted = TRUE )
str(la)

mu <- la$mu
str(mu)

# returns $iterations, $chains, $parameters
a <- extract( fit2, permuted = FALSE )
str(a)

a2 <- as.array( fit2 )
str(a2)

m <- as.matrix( fit2 )
str(m)

print( fit2, digits=1 )
