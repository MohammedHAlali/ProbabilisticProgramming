library(rstan)
library(ggplot2)

dso <- stan_model( stanc_ret=stanc(file="./LEGI-Amp.stan") )

ta <- 0.1;
ti <- 0.01;
tr <- 1.0;
tm <- 0.3;
kl <- 0.3;
kd <- 2.0;
kc <- 0.1;
rTot <- 1.0;
L <- 1.e2;
t.len <- 200;
ts <- seq(1, 200, length.out=t.len);

datalist <- list(
	T=t.len,
	y0=c( kc/kd, kc/kd, 0.5),
	t0=0,
	ts=ts,
	theta=c( ta, ti, tr, tr, kl, kd, kc, rTot, L)
)

fixed.sim <- sampling(dso,
	data=datalist,
	chains=1,
	iter=1,
	warmup=0,
	algorithm="Fixed_param"
)

y <- extract( fixed.sim, pars='y_hat' )$y_hat

str(y)

length(y[,1])
length(ts)

plot( ts, y[1,,2] )
