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


