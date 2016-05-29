# libs # {{{
library(grid)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rstan) 
options(dplyr.print_min=45)
options(max.print=450)
rstan_options(auto_write=TRUE)
# }}}

n <- 5

a <- rep( 1, n)

W <- diag(n)

s <- rbinom( n, 1, 0.5)

f <- function (s) {
  exp( t(a) %*% s + 0.5 * t(s) %*% W %*% s )
}

df <- expand.grid( rep( list(c(0,1)), n) ) %>% as.matrix() 

df[1,]

Z <- sum(apply(df, 1, f))

p <- function(s) f(s) / Z

p(c(1,1,1,1,1))

apply(df, 1, p) %>% sum()

library(MASS)

mvrnorm(10, W%*%s, W)






