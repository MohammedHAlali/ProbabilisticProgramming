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

# refs {{{

# https://ariddell.org/horseshoe-prior-with-stan.html
# http://andrewgelman.com/2015/02/17/bayesian-survival-analysis-horseshoe-priors/#comment-211738
# https://s3.amazonaws.com/docs.jrnold.me/Arnold_PolmethXXX_Poster.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=65C457F43CE140B0FD0FEBCA8E4BC96C?doi=10.1.1.217.2084&rep=rep1&type=pdf

# }}}

dso <- stan_model( stanc_ret=stanc(file="./HorseshoePrior.stan"))

datalist <- list( N=2
                 )
print(datalist)

# sampling {{{
chains <- 4
iter <- 2e3
warmup <- 1e3
samples <- NULL
samples <- sampling(dso
  , data=datalist
  , chains=chains
  , iter=warmup+iter
  , warmup=warmup
)

print(samples)

traceplot(samples)

pairs(samples, pars=c("lp__","w[1]","w[2]"
                      ,"lambda[1]","lambda[2]"))

df.stats <- monitor(samples,print=F) %>% as.data.frame() 
df.stats$params <- row.names(df.stats)
df.stats <- df.stats %>% separate( params, into=c("param", "ti")
    , sep = "[\\[\\]]", remove=TRUE, convert=TRUE, extra="drop") %>% 
    print()
df.stats %>% select( param, ti) %>% distinct(param)

# }}}

# plots {{{
df <- NULL
df <- as.data.frame(samples) %>% tbl_df() %>% 
  mutate(sample=factor(1:(chains*iter))) %>% 
    select( everything(), w_1=`w[1]`, w_2=`w[2]` ) %>% 
    print()


ggplot( data=df, aes(x=w_1, y=w_2)) +
  geom_point(aes(color=lp__),size=4,alpha=0.1)


ggplot( data=df, aes(x=w_1, y=w_2)) +
  geom_density2d()


qplot(c(0, 1)
      , stat = "function"
      , fun=function(x) 1/dgamma(x,.1,1)
      , geom = "line")

df %>% filter( -100 < w_1, w_1 < 100 ) %>% 
  qplot( x=w_1, data=., geom="histogram", binwidth=1)

# }}}

# Horseshoe+ prior {{{

dso.HorseshoePlusPrior <- NULL
dso.HorseshoePlusPrior <- stan_model( stanc_ret=stanc(file="./HorseshoePlusPrior.stan"))

datalist <- list( N=2 )
print(datalist)

# sampling {{{
chains <- 4
iter <- 2e3
warmup <- 1e3
samples <- NULL
samples <- sampling(dso.HorseshoePlusPrior
  , data=datalist
  , chains=chains
  , iter=warmup+iter
  , warmup=warmup
)

print(samples)

traceplot(samples)

pairs(samples, pars=c("lp__","w[1]","w[2]"
                      ,"lambda[1]","lambda[2]"))

df.stats <- monitor(samples,print=F) %>% as.data.frame() 
df.stats$params <- row.names(df.stats)
df.stats <- df.stats %>% separate( params, into=c("param", "ti")
    , sep = "[\\[\\]]", remove=TRUE, convert=TRUE, extra="drop") %>% 
    print()
df.stats %>% select( param, ti) %>% distinct(param)

# }}}

# plots {{{
df <- NULL
df <- as.data.frame(samples) %>% tbl_df() %>% 
  mutate(sample=factor(1:(chains*iter))) %>% 
    select( everything(), w_1=`w[1]`, w_2=`w[2]` ) %>% 
    print()

ggplot( data=df, aes(x=w_1, y=w_2)) +
  geom_point(aes(color=lp__),size=4,alpha=0.1)

ggplot( data=df, aes(x=w_1, y=w_2)) +
  geom_density2d()

df %>% filter( -100 < w_1, w_1 < 100 ) %>% 
  qplot( x=w_1, data=., geom="histogram", binwidth=1)

# }}}




# }}}
