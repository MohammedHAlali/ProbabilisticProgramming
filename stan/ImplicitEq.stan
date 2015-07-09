data {
}
transformed data {
}
parameters {
	real x; 
	real y;
}
model {
  0 ~ normal( x*x + y*y - 25 , 1 );
  0 ~ normal( y - x, 1 );
}
