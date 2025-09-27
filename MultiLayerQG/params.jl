# Parameters for two layer flows using GeophysicalFlows.jl

module Params

# include all modules
include("utils.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler

# local import
import .Utils



  	 ### Save path and device ###

# format: kappa_star = 0.1, Ktopo = 25, h_star = 1
expt_name = "/kappa01_kt25_h1"
path_name = "/scratch/mp6191/GeophysicalFlows_expts" * expt_name * "/output" * expt_name * ".jld2"

dev = GPU() # or CPU()

	 ### Grid ###

nx = 1024           # number of x grid cells
Ld = 15e3           # deformation radius
ld = 2 * pi * Ld    # deformation wavelength
Kd = 1 / Ld         # deformation wavenumber
Lx = 25 * ld        # side length of square domain

nz = 2                         # number of z layers
H0 = 4000.                     # total rest depth of fluid
delta = 1.                     # ratio of layer 1 to layer 2
H1 = delta / (1 + delta) * H0  # rest depth of layer 1
H2 = 1 / (1 + delta) * H0      # rest depth of layer 2
H = [H1, H2]                   # rest depth of each layer


    	 ### Control parameters ###

beta_star = 0.      # nondimensional β, β* = βλ^2/U
kappa_star = 0.1    # nondimensional κ, κ* = κλ/U
h_star = 5.         # nondimensional advection-topography, h* = f_0 * h_0 / (U * H * K_t)



    	 ### Planetary parameters ###

U0 = 0.01              						     # mean shear strength
U = [2 * U0, 0 * U0]   						     # velocity in each layer

f0 = 1e-4                                        # constant Coriolis
beta = U0 / Ld^2 * beta_star                     # y gradient of Coriolis

g = 9.81                                         # gravity
rho0 = 1000.                                     # reference density
rho1 = rho0 + 25.                                # layer 1 density
rho2 = rho1 / (1 - (4 * f0^2 * Ld^2) / (g * H0)) # layer 2 density given the deformation radius and other parameters
rho = [rho1, rho2]							     # density in each layer
# b = -g / rho0 .* [rho1, rho2]

kappa = 2 * U0 / Ld * kappa_star	             # linear Ekman drag, the factor of 2 is to be consistent with Gallet Ferrari (and Thompson Young)

      	   ### Topography ###
	   
Ktopo = 25 * (2 * pi / Lx)							        # topographic wavenumber
hrms = h_star * U0 * H0 * Ktopo / f0                        # rms topographic height 
h = Utils.monoscale_random(hrms, Ktopo, Lx, nx, dev)        # random monoscale topography
#h = Utils.goff_jordan_iso(h_star, f0, U0, H0, Lx, nx, dev) # random Goff Jordan topography
eta = f0 / H2 .* h                                          # bottom layer topographic PV



      	   ### Time stepping ###

Ti = Ld / U0               # nondimensional time
tmax = 600 * Ti            # final time in seconds
dt = 60 * 3.               # time step in seconds
dtsnap = 60 * 60 * 24 * 5  # snapshot frequency in seconds
nsubs = Int(dtsnap / dt)   # number of time steps between snapshots
nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs) * nsubs  # total number of model steps, nsubs should divide nsteps
stepper = "FilteredRK4"   # timestepper


			### Initial condition ###

K0 = 1 / (4 * Ld)          # most unstable Eady wavenumber, Km = 2 * pi / (4 * Ld) (see Vallis)
E0 = 1e-5                  # total average initial energy

end