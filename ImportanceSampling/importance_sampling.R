library(dplyr)
library(ggplot2)
library(rjulia)
julia_init()

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
