# libs {{{
library(gridExtra)
library(dplyr)
library(tidyr)
library(rstan)
library(shinystan)
rstan_options(auto_write=TRUE)
# }}}

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

# large matrix {{{

datalist <- list( 
    x1=runif(100)
  , x2=seq(0,1,0.05)
  , rho_sq=1e0
  , eta_sq=1e1
  , sigma_sq_y=1e-6
  , sigma_sq_w=1e-2
)
datalist$w1 <- (datalist$x1 - 0.5) + 
  rnorm(length(datalist$x1), mean=0, sd=0.5)
datalist$N1 <- length(datalist$x1)
datalist$N2 <- length(datalist$x2)
df.x2 <- data.frame( x2=datalist$x2, idx=1:length(datalist$x2))
qplot( x=datalist$x1, y=datalist$w1, geom="point", size=5)

iter <- 100
samps <- stan( "GP-deriv-large.stan"
              , data=datalist
              , chains=32
              , iter=iter+20
              , warmup=iter 
              , open_progress=FALSE
              , cores=4 )

print(samps)

traceplot( samps )

extract(samps) %>% as.data.frame() %>% tbl_df()

df.y2 <- extract( samps, pars=c("y2","w2","lp__")) %>% as.data.frame() %>% tbl_df()
df.y2 <- mutate( df.y2, sample=factor(1:nrow(df.y2)) ) %>% 
  gather( param, value, -lp__, -sample) %>% 
  separate( param, into=c("param", "idx"), convert=TRUE) %>%
  spread( param, value ) %>% 
  left_join( df.x2, by="idx") %>%
  print()

ggplot( data=df.y2, aes(x=x2, y=y2, group=sample, color=lp__)) +
  geom_line(alpha=0.4) +
  theme(legend.position="none")

qplot( x=datalist$x1, y=datalist$w1, geom="point", size=3) +
  geom_line( data=df.y2, aes(x=x2, y=w2, group=sample, color=lp__), alpha=0.05) +
  theme(legend.position="none")



# }}}

# large matrix - optimization {{{

datalist <- list( 
    x1=runif(80)
  , x2=seq(-2,2,0.1)
  , rho_sq=1e0
  , eta_sq=1e1
  , sigma_sq_y=1e-6
  , sigma_sq_w=5e-1
)
datalist$w1 <- (datalist$x1 - 0.5) + 
  rnorm( length(datalist$x1), sd=0.5)
datalist$N1 <- length(datalist$x1)
datalist$N2 <- length(datalist$x2)
df.x2 <- data.frame( x2=datalist$x2, idx=1:length(datalist$x2))
qplot( x=datalist$x1, y=datalist$w1, geom="point", size=5)

model <- stan_model( "GP-deriv-large.stan" )

opt <- optimizing( model
              , data=datalist
              , as_vector=FALSE
              , iter=1e4
              )
str(opt$par)

opt$par$x2=datalist$x2
opt$par$y2w2 <- NULL
df.opt <- as.data.frame(opt$par)
gg.y2 <- qplot( x=x2, y=y2, data=df.opt, geom="line")
gg.w2 <- 
  qplot( x=datalist$x1, y=datalist$w1, geom="point", size=8) +
  geom_line( data=df.opt, aes(x=x2, y=w2), size=1)
grid.arrange( gg.y2, gg.w2, ncol=1)

df.y2w2 <- data.frame(param=names(opt$par), value=opt$par) %>% 
  separate( param, into=c("param","idx"), convert=TRUE, extra="drop" ) %>% 
  spread( param, value ) %>% 
  select( -y2w2 ) %>% 
  na.omit() %>% 
  left_join( df.x2, by="idx") %>% 
  print()

ggplot( data=df.y2w2, aes(x=x2, y=y2)) +
  geom_line(alpha=0.4) +
  theme(legend.position="none")

qplot( x=datalist$x1, y=datalist$w1, geom="point") +
  geom_line( data=df.y2w2, aes(x=x2, y=w2), alpha=0.5) +
  theme(legend.position="none")




# }}}
