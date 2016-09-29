
julia> Profile.clear_malloc_data()

julia> @time negloglikelihood( q_init )
  0.001466 seconds (3.90 k allocations: 153.406 KB)
309.9949633031291

julia> @time SPSAmod.searchADAM( spsa, 100 )
  0.754610 seconds (4.01 M allocations: 156.877 MB, 3.51% gc time)
18-element Array{Float64,1}:
  1.31872
  1.23615
  2.17488
  1.77261
  1.31604
  2.15014
  0.44546
  1.54081
  0.511135
  0.265434
  0.693268
  0.511701
  0.206149
  1.06318
  1.1829
  0.199244
  0.162426
 -0.564658

julia>













































