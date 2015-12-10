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

 

qplot(c(0, 10), stat = "function", fun=function (x) dgamma( x, 2, scale=1/10), geom = "line")
