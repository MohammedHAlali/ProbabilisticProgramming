using Distributions


function cov( σ, ρ) 
  Σ = Float64[ σ^2 ρ*σ^2 ; ρ*σ^2 σ^2 ]
end

idp = MvNormal(cov(1,0))
entropy(idp)

mvn = MvNormal(cov(1,.999))
KL(mvn,idp)

ρs = -0.9999:0.01:0.9999
dat = [KL(MvNormal(cov(1,ρ)),idp) for ρ=ρs]
plot(ρs,dat)

entropy(mvn)

function crossentropy( p, q )
  samp = rand(p, 100)
  ce = logpdf(q, samp) |> mean
  return -ce
end


function KL(p, q)
  crossentropy(p,q) - entropy(p)
end

det( cov(1,1) )

dat = rand(mvn, 100)
scatter(dat[1,:],dat[2,:])


