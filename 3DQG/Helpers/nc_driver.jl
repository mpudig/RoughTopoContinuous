# Convert jld2 file to nc file
cd("..")
dir = pwd()
include(dir * "/Helpers/jld2_to_nc.jl")
convert_to_nc_fields()
convert_to_nc_diags()
