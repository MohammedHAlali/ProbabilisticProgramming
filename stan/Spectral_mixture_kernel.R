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

# Gaussian Process Kernels for Pattern Discovery and Extrapolation
# http://jmlr.org/proceedings/papers/v28/wilson13.pdf

phi <- function(s, mu=1, sig=1)
  exp( -(s-mu)^2/(2*sig^2) ) / sqrt(2*pi*sig^2)

phi(0)

qplot(c(-4,4), stat = "function", fun=phi, geom = "line")


SpectralDensity <- function(s, mu=2, sig=.8) (phi(s,mu,sig)+phi(-s,mu,sig)) / 2

qplot(c(-4,4), stat = "function", fun=SpectralDensity, geom = "line")

k <- function(tau, mu=1, sig=1) exp(-2*pi^2*tau^2*sig^2)*cos(2*pi*tau*mu)

qplot(c(-4,4), stat = "function", fun=k, geom = "line", args=list(mu=.1, sig=2))


