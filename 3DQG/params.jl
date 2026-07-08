### Parameters for 3D QG turbulence simulations driven by mean fields/topography from GLORYS12v1 reanalysis ###

module Params

# include all modules
dir = pwd()
include(dir * "/Helpers/utils.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, SpecialFunctions, ForwardDiff, JLD2, NCDatasets

# local import
import .Utils

		### Save path and device ###
root = "/scratch/mp6191/RoughTopoContinuous/Glorys"
expt_name = "/region"

restart_num = 0
if restart_num == 0
path_name = root * expt_name * "/output" * expt_name * ".jld2"
else
path_name = root * expt_name * "/output" * expt_name * "_restart$restart_num" * ".jld2"
end

dev = GPU() # or CPU()

        ### Read in mean fields ###
mean_fields = NCDataset(root * expt_name * "/input" * "/mean_fields.nc", "r")

# Fields
U = Float64.(mean_fields["U"][:])
V = Float64.(mean_fields["V"][:])
N² = Float64.(mean_fields["N2"][:])
h = Float64.(mean_fields["h"][:, :])

# Coordinates
x = Float64.(mean_fields["x"][:])
y = Float64.(mean_fields["y"][:])
z = Float64.(mean_fields["z"][:])

# Attribs
f₀ = Float64(mean_fields.attrib["f0"])
β = Float64(mean_fields.attrib["beta"])
H₀ = Float64(mean_fields.attrib["H0"])
Ld = Float64(mean_fields.attrib["Ld"])

		### Resolution ###

nx = length(x)       # number of x, y grid points
nz = length(z)       # number of z grid points

    	### Domain scalar parameters ###

Lx = x[end]            # square domain side length
cd = 0.003 		       # quadratic drag used by model

      	### Time stepping ###

Umax = max(maximum(U), maximum(V))    # rough magnitude of mean shear
Ti = Ld / Umax                        # nondimensional time
tmax = 300 * Ti                       # final time [s]
dt = 60 * 60                          # initial time step [s]

dtsnap_diags = 5 * 86400             # snapshot frequency for diagnostics [s]
dtsnap_fields = 15 * 86400           # snapshot frequency for fields [s]

nsubs_diags = Int(floor(dtsnap_diags / dt))       # number of time steps between snapshots for saving diagnostics
nsubs_fields = Int(floor(dtsnap_fields / dt))     # number of time steps between snapshots for saving fields

nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs_fields) * nsubs_fields    # total number of model steps, nsubs_fields > nsubs_diags so this defines the total number of model time steps

stepper = "FilteredRK4"              # timestepper


			### Initial condition ###

K0 = 1 / (4 * Ld)                                 # most unstable Eady wavenumber, Km = 2 * pi / (4 * Ld) (see Vallis text)
E0 = 1e-3                                         # initial energy [m2 s-2]
ϕ₁ = Utils.first_baroclinic_mode(f₀, H₀, N², nz)  # first baroclinic mode                         

close(mean_fields)
end
