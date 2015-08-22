library(dplyr)
library(tidyr)
library(rstan)
library(shinystan)
rstan_options(auto_write=TRUE)

stan( "GP-deriv.stan" )

install.packages( "shinystan" )

launch_shinystan_demo()

datalist <- list( 
    x1=c(-1,0,1)
  , y1=c(1, 1,1)
  , x2=seq(-2,2,0.1)
  , rho_sq=1e-2
  , eta_sq=1
  , sigma_sq=1e-6
)
datalist$N1 <- length(datalist$x1)
datalist$N2 <- length(datalist$x2)
df.x2 <- data.frame( x2=datalist$x2, idx=1:length(datalist$x2))

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

