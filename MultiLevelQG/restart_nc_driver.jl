# Convert jld2 file to nc file
include("restart_jld2_to_nc.jl")
convert_to_nc_fields()
convert_to_nc_diags()