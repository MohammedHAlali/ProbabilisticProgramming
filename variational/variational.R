# libs # {{{
library(grid)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rjulia)
julia_init()
options(dplyr.print_min=45)
options(max.print=450)
# }}}

# simple N(u,s) ~ Q(u,s) {{{

j2r('include("./variational.jl")')

j2r('s = SPSAmod.SPSA( [0.,0.], variational.lossFunction )')
j2r('s.dQmax=1.0')
j2r("SPSAmod.searchRestart(s,5)")
j2r('s.dQmax=0.1')
j2r("SPSAmod.searchFinal(s)")

j2r("variational.lossFunction(s.Qold)")
j2r("s.Qold")

df <- j2r("s.param_history") %>% tbl_df() %>% 
  mutate( logloss = log(loss)
         , logak = log(ak)
         , logck = log(ck)
         , M = factor(M) 
         , simNum = factor(simNum) ) %>% 
  gather( param, value, -iter, -simNum, -M )

ggplot( data=df, aes(x=iter, y=value)) +
  facet_wrap( ~param, scales="free_y", ncol=2 ) +
  theme(legend.position="none") +
  geom_point(aes(color=simNum, group=M),size=.1) #+ xlim(c(-1,3)+1000)

'SPSAmod.clear_history(s)' %>% j2r()

# }}}

# normal approx {{{

len <- j2r('variational.s.iter')
idx <- round(seq.int(1,len,length.out=21))

df <- j2r('variational.posteriorPDF()') %>% mutate(lp=exp(lp))

for (i in idx) {
  df.v <- j2r(paste('variational.posteriorVariationalPDF(
            convert(Array, variational.s.param_history[',i,',[:x1,:x2,:x3,:x4]] 
            ) |> vec)')) %>% tbl_df() %>% 
  mutate(lp=exp(lp)) 
print(ggplot( data=df, aes(x=mu, y=sd, fill=lp)) +
    geom_raster() +
    geom_contour(data=df.v,aes(z=lp)) 
)
}


df.v <- j2r('variational.posteriorVariationalPDF(variational.s.Qold)') %>% 
  mutate(lp=exp(lp))
ggplot( data=df, aes(x=mu, y=sd, fill=lp)) +
  geom_raster() +
  geom_contour(data=df.v,aes(z=lp))

  scale_fill_gradient(limits=c(-166,-106))

elbo <- j2r('variational.elbo()')

qplot( x=mu, y=elbo, data=elbo, geom="line")

df = expand.grid(x=1:3, y=1:4)

ggplot( data=df, aes(x=x, y=energy)) +
  geom_line()

df <- data.frame( x = seq(0,200,length.out=100) ) %>% 
  mutate( gamma = dgamma(x, 8, 10/1) )
qplot( x=x, y=gamma, data=df, geom="line")


ggplot( data=faithful, aes(x=waiting, y=eruptions, z=density)) +
  geom_contour()

faithful %>% mutate( z = waiting^2 + eruptions^2 ) %>% 
ggplot( aes(waiting,eruptions,z=z))+
  geom_raster(fill=density) +
  geom_point()

ggplot(faithful,aes(waiting,eruptions)) + 
  geom_raster( aes(fill=density) )
  # geom_raster( aes(fill=density), hjust=0.5, vjust=0.5, interpolate=FALSE)

faithfuld %>% 
ggplot(aes(waiting, eruptions, density)) +
  geom_contour(aes(fill=density))

  geom_raster(aes(fill=density),interpolate=T)

  geom_raster(aes(fill = density), interpolate = FALSE)


faithfuld %>% sample_n(10)

# }}}

# mixture models {{{

library(ggplot2)
library(dplyr)
library(tidyr)

df <- df_data %>% tbl_df() %>% 
  mutate( z=factor(z) ) %>% 
  print()

df_mu <- df_mus %>% tbl_df() %>% 
  mutate( simNum = ordered(simNum) ) %>%
  rename( x1.1 = x1
         ,x2.1 = x2
         ,x1.2 = x3
         ,x2.2 = x4
         ,x1.3 = x5
         ,x2.3 = x6
         ) %>% 
  gather( param, value, -iter, -simNum, -M ) %>% 
  separate(param, c("v1","z")) %>% 
  spread( v1, value)

df.zp <- j2r("MixtureModels.getZp()") %>% tbl_df() %>% 
  gather( param, value, -iter, -simNum, -M)

ggplot( data=df, aes(x=x1, y=x2, color=z)) +
  geom_point(size=8) +
  geom_path( data=df_mu %>% filter(simNum==0), size=1, aes())

df_mus %>% select(iter, M,simNum)

$(MixtureModels.getEllipse(model, spsa, 1, 1))

ellip <- function(idx, z) {
  geom_path(data=j2r(paste('MixtureModels.getCov(',idx,',',z,')'))
            %>%mutate(z=factor(z)), aes(x=x1,y=x2))
}
ggplot(data=df, aes(x=x1,y=x2,color=z)) +
  geom_point(size=8) +
  geom_path( data=df_mu %>% filter(M==4), size=1, aes()) +
  # ellip(1,1) + ellip(1,2) + ellip(1,3) +
  # ellip(10000,1) + ellip(10000,2) + ellip(10000,3) +
  ellip(idx,1) + ellip(idx,2) + ellip(idx,3)


j2r('MixtureModels.spsa.param_history |> size')

df %>% count(z)

df.param <- j2r("MixtureModels.spsa.param_history") %>% tbl_df() %>% 
  mutate( logloss = log(loss)
         , logak = log(ak)
         , logck = log(ck)
         , M = factor(M) 
         , simNum = factor(simNum) ) %>% 
  gather( param, value, -iter, -simNum, -M )

ggplot( data=df.param, aes(x=iter, y=value)) +
  facet_wrap( ~param, scales="free_y", ncol=2 ) +
  theme(legend.position="none") +
  geom_point(aes(color=simNum, group=M),size=.1) #+ xlim(c(-1,3)+1000)

df.param %>% filter( param=="loss" ) %>% 
  tail(8000) %>% 
ggplot( data=. ) +
  geom_line(aes(x=iter,y=value,color=simNum,group=M))

ggplot( data=df.zp, aes(x=iter, y=value)) +
  facet_wrap( ~param, scales="free_y", ncol=3 ) +
  theme(legend.position="none") +
  geom_point(aes(color=simNum, group=M),size=.1) #+ xlim(c(-1,3)+1000)


df.prof <- read.table('/tmp/prof.txt', header=TRUE) %>% tbl_df() %>% 
  arrange(desc(Count)) %>% 
  mutate(Line=factor(Line),
         Function=factor(Function,Function),
         Frac=Count/max(Count)) %>% 
  filter(Frac>0.25)
head(df.prof)
qplot( x=Frac, y=Function, data=df.prof, color=Line, geom="point", size=10)

ggplot( data=%.%, aes(x=time, y=value)) +


# }}}


