library(dplyr)
library(tidyr)
library(rstan)
library(shinystan)
rstan_options(auto_write=TRUE)

# predictions of w at x1 only {{{
stan( "GP-deriv.stan" )

install.packages( "shinystan" )

launch_shinystan_demo()

datalist <- list( 
    x1=c(-1,0,1, 0, 0)
  , y1=c( 1,1,1, 0.9, 1.1)
  , x2=seq(-2,2,0.1)
  , rho_sq=1e-2
  , eta_sq=1
  , sigma_sq_y=1e-6
  , sigma_sq_w=1e1
)
datalist$N1 <- length(datalist$x1)
datalist$N2 <- length(datalist$x2)
df.x2 <- data.frame( x2=datalist$x2, idx=1:length(datalist$x2))
qplot( x=datalist$x1, y=datalist$y1, geom="point", size=5)

samps <- stan( "GP-deriv.stan", data=datalist )

launch_shinystan( samps )

traceplot( samps )

extract(samps) %>% as.data.frame() %>% tbl_df()

df.y2 <- extract( samps, pars=c("y2","lp__")) %>% as.data.frame() %>% tbl_df()
df.y2 <- mutate( df.y2, sample=factor(1:nrow(df.y2)) ) %>% 
  gather( param, value, -lp__, -sample) %>% 
  separate( param, into=c("param", "idx"), convert=TRUE) %>%
  spread( param, value ) %>% 
  left_join( df.x2, by="idx") %>%
  print()

ggplot( data=df.y2, aes(x=x2, y=y2, group=sample, color=lp__)) +
  geom_line(alpha=0.4) +
  theme(legend.position="none")

dso <- stan_model( stanc_ret=stanc(file="GP-deriv.stan" ) )

opt <- optimizing(dso, data=datalist )
# }}}


# concatentated matricies {{{

datalist <- list( 
    x1=c(-1,0,1, 0, 0)
  , y1=c( 1,1,1, 0.9, 1.1)
  , x2=seq(-2,2,0.1)
  , rho_sq=1e-2
  , eta_sq=2
  , sigma_sq_y=1e-6
  , sigma_sq_w=1e-6
)
datalist$N1 <- length(datalist$x1)
datalist$N2 <- length(datalist$x2)
df.x2 <- data.frame( x2=datalist$x2, idx=1:length(datalist$x2))
qplot( x=datalist$x1, y=datalist$y1, geom="point", size=5)

samps <- stan( "GP-deriv-concat.stan", data=datalist )

print(samps)

traceplot( samps )

extract(samps) %>% as.data.frame() %>% tbl_df()

df.y2 <- extract( samps, pars=c("y","w","lp__")) %>% as.data.frame() %>% tbl_df()
df.y2 <- mutate( df.y2, sample=factor(1:nrow(df.y2)) ) %>% 
  gather( param, value, -lp__, -sample) %>% 
  separate( param, into=c("param", "idx"), convert=TRUE) %>%
  spread( param, value ) %>% 
  left_join( df.x2, by="idx") %>%
  print()

ggplot( data=df.y2, aes(x=x2, y=y, group=sample, color=lp__)) +
  geom_line(alpha=0.4) +
  theme(legend.position="none")

qplot( x=datalist$x1, y=datalist$y1, geom="point") +
  geom_line( data=df.y2, aes(x=x2, y=w, color=sample), alpha=0.05) +
  theme(legend.position="none")


# }}}
