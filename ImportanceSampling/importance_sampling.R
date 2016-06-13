library(dplyr)
library(tidyr)
library(ggplot2)
library(rjulia)
julia_init()

# importance_sampling.jl {{{
j2r('include("./importance_sampling.jl")')
df <- j2r('Ex34MonteCarloIntegration.df') %>% as.matrix() %>% data.frame()
str(df)

mean(df$hu)

df$hm[10000]

var( df$hu )

df$v[10000]

sum((df$hu - df$hm[10000])^2) / (10000)

qplot( x=x, y=h, data=df,  geom="point") +
  theme(legend.position="none")

qplot( df$hu )


filter(df, m>1) %>% 
ggplot() +
  geom_hline( yintercept=0.965, size=2, color='green', alpha=0.5) +
  geom_ribbon(aes(x=m, ymin=hm-se2,ymax=hm+se2),alpha=0.1,fill='red') + 
  geom_line(aes(x=m, y=hm)) +
  ylim(0.9,1.1)

filter(df, m>3) %>% 
qplot( x=m, y=se, data=., geom="line", color='red')

filter(df, m>10) %>% 
qplot( x=m, y=sqrt(v), data=., geom="line") 
# }}}

# Ex38CauchyTailProb.jl {{{

j2r('include("./Ex38CauchyTailProb.jl")')
df <- j2r('Ex38CauchyTailProb.df') %>% as.matrix() %>% data.frame() %>% tbl_df()
df.avg <- j2r('Ex38CauchyTailProb.df_avg') 

df %>% gather( h, value ) %>% 
  ggplot( data=., aes(x=value)) +
    facet_wrap( ~h, scales="free_y", ncol=1 ) +
    geom_histogram(bins=300) +
    geom_vline( data=df.avg, aes(xintercept=mean), size=2, color='red', alpha=0.5) +
    geom_vline( data=df.avg, aes(xintercept=mean+var), size=1, color='black', alpha=0.5) +
    geom_vline( data=df.avg, aes(xintercept=mean-var), size=1, color='black', alpha=0.5)

summary(df)

qplot( x=h3, data=df, geom="histogram")

ggplot(data.frame(x=c(0,2),y=c(0,0.5))) +
  stat_function(aes(x),fun=function(x) 2/(pi*(1+x^2)),color=1) +
  stat_function(aes(y),fun=function(x) x^-2/(pi*(1+x^-2)),color=2)

df.iter <- j2r('Ex38CauchyTailProb.df_iter') %>% as.matrix() %>% data.frame() %>% tbl_df()

ggplot(df.iter) +
  geom_hline( yintercept=0.15, size=2, color='green', alpha=0.5) +
  geom_ribbon(aes(x=m, ymin=m1-se1,ymax=m1+se1),alpha=0.2,fill='red') + 
  geom_ribbon(aes(x=m, ymin=m2-se2,ymax=m2+se2),alpha=0.2,fill='blue') + 
  geom_ribbon(aes(x=m, ymin=m3-se3,ymax=m3+se3),alpha=0.2,fill='green') + 
  geom_ribbon(aes(x=m, ymin=m4-se4,ymax=m4+se4),alpha=0.2,fill='black') +
  ylim(2e-2*c(-1,1)+0.15)

# }}}
