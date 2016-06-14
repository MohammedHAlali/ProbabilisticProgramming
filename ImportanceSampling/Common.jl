module Common #{{{

export onlineMeanVar

function onlineMeanVar(x::Array{Float64,1})
  M = length(x)
  mean = zeros(x)
  var = zeros(x)
  mean[1] = x[1]
  M2 = 0
  for m = 2:M
    delta = x[m] - mean[m-1]
    mean[m] = mean[m-1] + delta / m
    M2 += delta * (x[m] - mean[m])
    var[m] = M2 / m
  end
  return (mean, var)
end

end # Common }}}

