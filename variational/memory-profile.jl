module profiling
using Distributions
include("./variational.jl")
include("/home/abergman/projects/optimization/SPSA.jl")

function profile_search()
  model = MixtureModels.MixtureModel(3,2)
  df = MixtureModels.GenerateData()
  X = convert(Array,df[[:x1,:x2]]) 
  Q = MixtureModels.init_Q(model, X) 
  nll = MixtureModels.Gen_negloglikelihood( model, X )
  spsa = SPSAmod.SPSA( Q, nll)
  SPSAmod.clear_history(spsa,true)
  spsa.dQmax = 0.001
  sol = SPSAmod.searchADAM( spsa, 10, 1)

  Profile.clear_malloc_data()
  sol = SPSAmod.searchADAM( spsa, 10000, 1)
  print( hash(sol) )
end 


function profile_negloglikelihood()
  model = MixtureModels.MixtureModel(3,2)
  df = MixtureModels.GenerateData()
  X = convert(Array,df[[:x1,:x2]]) 
  Q = MixtureModels.init_Q(model, X) 

  k = model.k
  N = size(X,1)
  model.lH = zeros(k)
  model.lp_mvn = [zeros(N) for i=1:k]
  model.lp_z = zeros(k)
  sol = MixtureModels.negloglikelihood( model, X, Q )

  Profile.clear_malloc_data()
  sol = MixtureModels.negloglikelihood( model, X, Q )
  print( hash(sol) )
end # function profile_negloglikelihood

function profile_logpdf()
  N = 1000
  m = 100
  mvndist = MvNormal(ones(m),ones(m,m)+eye(m))
  x = rand(m,N)
  lp = zeros(N)
  logpdf!(lp, mvndist, x)
  @time logpdf!(lp, mvndist, x)
  Profile.clear_malloc_data()
  logpdf!(lp, mvndist, x)
end



profile_logpdf()

end
