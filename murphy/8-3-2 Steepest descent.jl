using Plots
include("/home/abergman/projects/optimization/SPSA.jl")

function SPSAmod.loss( θ )
  0.5*(θ[1]^2 - θ[2])^2 + 0.5*(θ[1]-1)^2
end


θ = zeros(2)
spsa = SPSAmod.SPSA( θ )
SPSAmod.clear_history(spsa, true)
spsa.kd=1000
spsa.dQmax=0.1
SPSAmod.searchADAM(spsa, 1000, 1)
plot(spsa)

dat = hcat(spsa.param_history...)
scatter(dat[1,:], dat[2,:],xlim=1+[-1,1]*1e-1,ylim=1+[-1,1]*1e-1)

plot(drop(dat[1,:],400)|>collect)

drop
