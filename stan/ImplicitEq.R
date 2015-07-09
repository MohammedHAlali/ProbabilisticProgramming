library(rstan)
library(ggplot2)
library(dplyr)

dso <- stan_model( stanc_ret=stanc(file="ImplicitEq.stan") )

fit <- sampling( 
  dso,
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

ggplot( s, aes(x=x, y=y)) +
  geom_point()

