### Parameters for multi-level QG turbulence simulations with small-scale topography and linear stratification ###

module Params

# include all modules
include("utils.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler

# local import
import .Utils

		### Save path and device ###

# format: nz = ..., kappa = ..., h = ...
expt_name = "/nz32_r02_h5"
path_name = "/scratch/mp6191/RoughTopoContinuous/LinStrat" * expt_name * "/output" * expt_name * ".jld2"

dev = GPU() # or CPU()

		### Resolution ###

nx = 512             # number of x, y grid points
nz = 12              # number of z grid points

    	### Control parameters ###

r_star = 0.2		  # nondimensional drag coefficient, r* = f₀λr/UH
h_star = 1.           # nondimensional advection-topography, h* = f₀h₀/UHKₜ
β_star = 0.			  # nondimensional beta, β* = βλ²/U

		### Domain ###

Ld = 20e3           # first baroclinic deformation radius [m]
Kd = 1 / Ld         # first baroclinic deformation wavenumber [m-1]
ld = 2 * pi * Ld    # first baroclinic deformation wavelength [m]
Lx = 15 * ld        # side length of square domain [m]

H₀ = 4000.                      # total mean depth [m]
Hᵢ = H₀ / nz                    # equal height of each layer [m]
H =  Hᵢ .* ones(nz)             # array of layer heights
z = range(0., -H₀, nz + 1)      # vertical cell edges
zc = z[1 : end - 1] .- H ./ 2   # vertical cell centres


    	### Background scalar parameters ###

U₀ = 1e-2              			    # baroclinic shear [m s-1]
f₀ = 1e-4                           # constant Coriolis [s-1]
β = U₀ * β_star / Ld^2			    # y gradient of Coriolis [m-1 s-1]
N₀ = Utils.LinStrat(f₀, H₀, Ld)	    # buoyancy frequency magnitude for given deformation radius, etc [s-1]
r = U₀ * H₀ / (f₀ * Ld) * r_star    # linear drag [m]

		### Background profiles ###

ϕ₁ = sqrt(2) * cos.(N₀ / (Ld * f₀) * [0.; zc; -H₀]) # first baroclinic vertical mode at interior and surfaces
U = U₀ .* ϕ₁ .- (U₀ * ϕ₁[end]) 	                    # background zonal shear projected onto first baroclinic mode (with barotropic shift)
N² = N₀^2 .* ones(nz + 1)             		        # background constant N₀^2 at half levels [s-2]

      	### Topography ###

Ktopo = Kd															# minimum topographic wavenumber [m-1]
h = Utils.GoffJordanTopo(h_star, f₀, U₀, H₀, Ktopo, Lx, nx, dev)	# random Goff Jordan topography [m]

      	### Time stepping ###

Ti = Ld / U₀                							    # nondimensional time
tmax = 400 * Ti          						            # final time [s]
dt = 60 * 60 * 12                                           # time step [s]

dtsnap_diags = Ti / 5    						# snapshot frequency for diagnostics [s]
dtsnap_fields = dtsnap_diags							# snapshot frequency for fields [s]

nsubs_diags = Int(floor(dtsnap_diags / dt))     					# number of time steps between snapshots for saving diagnostics
nsubs_fields = Int(floor(dtsnap_fields / dt))     					# number of time steps between snapshots for saving fields

nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs_fields) * nsubs_fields    # total number of model steps, nsubs_fields > nsubs_diags so this defines the total number of model time steps

stepper = "FilteredRK4"   								    # timestepper


			### Initial condition ###

K0 = 1 / (4 * Ld)          # most unstable Eady wavenumber, Km = 2 * pi / (4 * Ld) (see Vallis text)
E0 = 1e-5                  # total average initial energy [m2 s-2]

end
