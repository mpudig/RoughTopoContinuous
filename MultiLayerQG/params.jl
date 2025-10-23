### Parameters for multi-layer QG turbulence simulations with small-scale topography and linear stratification ###

module Params

# include all modules
include("utils.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler

# local import
import .Utils

		### Save path and device ###

# format: nz = ..., kappa = ..., h = ...
expt_name = "/nz4_r02_h0"
path_name = "/scratch/mp6191/RoughTopoContinuous/LinStrat" * expt_name * "/output" * expt_name * ".jld2"

dev = GPU() # or CPU()

		### Resolution ###

nx = 512             # number of x, y grid points
nz = 4               # number of z grid points

    	### Control parameters ###

r_star = 0.2		  # nondimensional drag coefficient, r* = rf₀λ/UH
h_star = 0.           # nondimensional advection-topography, h* = f₀h₀/UHKₜ
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
g = 9.81                            # gravity [m2 s-1]
N₀ = Utils.LinStrat(f₀, H₀, Ld, nz)	# buoyancy frequency magnitude for given deformation radius, etc [s-1]
r = H₀ * U₀ / (f₀ * Ld) * r_star    # linear drag [m]
μ = f₀ / H[end] * r 			    # bottom layer drag [s-1]


		### Background profiles ###
b = N₀^2 .* zc             				    # background buoyancy profile given constant N₀ [m s-2]
F = Utils.calcF(f₀, b, H, nz)						# stretching matrix 
ϕ₁ = eigvecs(-F)[:,2]						# first baroclinic vertical mode
U = abs.(U₀ .* ϕ₁ .- (U₀ * ϕ₁[end]))        # background zonal shear projected onto first baroclinic mode (with barotropic shift so no background flow in lowest layer)

      	### Topography ###

Ktopo = Kd															# minimum topographic wavenumber [m-1]
h = Utils.GoffJordanTopo(h_star, f₀, U₀, H₀, Ktopo, Lx, nx, dev)	# random Goff Jordan topography [m]
eta = f₀ / H[end] .* h                                      		# bottom layer topographic PV [s-1]

      	### Time stepping ###

Ti = Ld / U₀                							    # nondimensional time
tmax = 400 * Ti          						            # final time [s]
dt = 60 * 60 * 12.                                          # time step [s]

dtsnap_diags = 60 * 60 * 24 * 30    						# snapshot frequency for diagnostics [s]
dtsnap_fields = 10 * dtsnap_diags							# snapshot frequency for fields [s]

nsubs_diags = Int(dtsnap_diags / dt)     					# number of time steps between snapshots for saving diagnostics
nsubs_fields = Int(dtsnap_fields / dt)     					# number of time steps between snapshots for saving fields

nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs_fields) * nsubs_fields    # total number of model steps, nsubs_fields > nsubs_diags so this defines the total number of model time steps

stepper = "FilteredRK4"   								    # timestepper


			### Initial condition ###

K0 = 1 / (4 * Ld)          # most unstable Eady wavenumber, Km = 2 * pi / (4 * Ld) (see Vallis text)
E0 = 1e-5                  # total average initial energy [m2 s-2]

end
