# Some useful extra functions for running my experiments with GeophysicalFlows.jl

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler
using FourierFlows: parsevalsum

"""
	MonoscaleTopo(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function MonoscaleTopo(hrms, Ktopo, Lx, nx, dev)

	# Wavenumber grid
	nk = Int(nx / 2 + 1)
	nl = nx
	
	dk = 2 * pi / Lx
	dl = dk
	 
	k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	K = @. sqrt(k^2 + l^2)

	# Isotropic Gaussian in wavenumber space about mean, Ktopo, with standard deviation, sigma
	# with random Fourier phases
	sigma = sqrt(2) * dk

	Random.seed!(1234)
	hh = exp.(-(K .- Ktopo).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))

	# Recover h from hh
	h = irfft(hh, nx)

	c = hrms / sqrt.(mean(h.^2))
	h = c .* h

	return h
end

"""
        GoffJordanTopo(h_star, f0, U0, H0, Ktopo, Lx, nx, dev)

Returns a 2D, isotropic topography field defined by the Goff Jordan spectrum with random phases.
"""

function GoffJordanTopo(h_star, f0, U0, H0, Ktopo, Lx, nx, dev)

    # Wavenumber grid
    nk = Int(nx / 2 + 1)
    nl = nx

    dk = 2 * pi / Lx
    dl = dk

    k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
    l = reshape( fftfreq(nx, dl * nx), (1, nl) )

    K = @. sqrt(k^2 + l^2)

    # Goff Jordan spectrum assuming isotropy, bandpass filtered for Ktopo < K < K_c * Ktopo
	# Kmin = Ktopo sets peak of isotropic power spectrum
	# Kmax = pi / (3*dx) (where dx = Lx / nx), based on having at least 6 grid points to resolve the smallest scale
	Kmin = Ktopo
	Kmax = pi / (3 * (Lx / nx))

	mu = 3.5
    k0 = Kmin * sqrt(mu - 1)
    l0 = k0

    Random.seed!(1234)
    hspec = @. (1 + (k / k0)^2 + (l / l0)^2)^(-mu / 2)
    hpf = ifelse.(K .> Kmin, K ./ K, 0 .* K)
    lpf = ifelse.(K .< Kmax, K ./ K, 0 .* K)
    CUDA.@allowscalar lpf[1, 1] = 1.
    bpf = lpf .* hpf
    hh = bpf .* sqrt.(hspec) .* exp.(2 * pi * im .* rand(nk, nl))

    # Recover h from hh
    h = irfft(hh, nx)

	# Get hrms for given h_star, and scale h
    hrms = h_star * U0 * H0 * Ktopo / f0
    c = hrms / sqrt.(mean(h.^2))
    h = c .* h

    return h
end

"""
        set_initial_condition!(prob, grid, K0, E0, ϕ₁)

        Sets the initial condition of MultiLayerQG to be a random q(x,y) field with baroclinic structure ϕ₁
        and with energy localized in spectral space about K = K₀ and with total energy equal to E₀
"""
function set_initial_condition!(prob, K0, E0, ϕ₁)
    params = prob.params
    grid = prob.grid
    vars = prob.vars
    dev = grid.device
    T = eltype(grid)
    A = device_array(dev)

    # Grid
    nx = grid.nx
    Lx = grid.Lx
    nz = params.nlevels

    nk = Int(nx / 2 + 1)
    nl = nx

    dk = 2 * pi / Lx
    dl = dk

    k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
    l = reshape( fftfreq(nx, dl * nx), (1, nl) )

    K2 = @. k^2 + l^2
    K = @. sqrt(K2)

    # Isotropic Gaussian in wavenumber space about mean, K0, with standard deviation, sigma
    sigma = sqrt(2) * dk

	Random.seed!(4321)
    psihmag = exp.(-(K .- K0).^2 ./ (2 * sigma^2)) .* exp.(2 * pi * im .* rand(nk, nl))

    # Give initial condition first baroclinic interior structure and zero surface buoyancy
    psih = zeros(nk, nl, nz + 2) .* im
    for j = 1 : nz
        psih[:, :, j + 1] = psihmag .* ϕ₁[j]
    end
    CUDA.@allowscalar psih[:, :, 1] = psih[:, :, 2]          # \partial_z \phi|z=0 ~ \phi_0 - \phi_1 
    CUDA.@allowscalar psih[:, :, end] = psih[:, :, end - 1]  # \partial_z \phi|z=-H ~ \phi_N - \phi_N+1 

    # Calculate KE and APE, and prescribe mean total energy
    # For simplicity, neglect the z = 0 and z = -H surfaces in the calculation
    H = params.H
    V = grid.Lx * grid.Ly * sum(H)
    f0, N² = params.f₀, params.N²

    abs²∇𝐮h = zeros(nk, nl, nz) .* im
    for j = 1 : nz
        abs²∇𝐮h[:, :, j] = K2 .* abs2.(psih[:, :, j + 1])
    end

    KE, PE = zeros(nz), zeros(nz - 1)

    for j = 1 : nz
        KE[j] = 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * params.H[j]
    end

	for j = 1 : nz - 1
        PE[j] = 1 / (2 * V) * params.f₀^2 ./ params.N²[j + 1] .* parsevalsum(abs2.(view(psih, :, :, j + 1) .- view(psih, :, :, j + 2)), grid)
    end

    E = sum(KE) + sum(PE)
    c = sqrt(E0 / E)
    psih = @. c * psih
    psih = A(psih)

    # Invert psih to get qh, then transform qh to real space qh
    qh = A(zeros(nk, nl, nz + 2)) .* im
    MultiLevelQG.pvfromstreamfunction!(qh, psih, params, grid)

    q = A(zeros(nx, nx, nz + 2))
    MultiLevelQG.invtransform!(q, qh, params)

    # Set as initial condition
    MultiLevelQG.set_q!(prob, q)
end

"""
        LinStrat(f₀, H, Ld)

Returns linear stratification for given first baroclinic deformation radius
"""

function LinStrat(f₀, H₀, Ld)
	N₀ = pi * f₀ * Ld / H₀

	return N₀
end


### Diagnostics ###

# Note:
# Most of these diagnostics assume constant stratification; they must be modified when using nonconstant stratification
# They also assume equal layer depths for simplicity

function BarotropicEKE(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

	nz = params.nlevels					    # number of interior levels
	ϕ₀ = A(reshape(ones(nz + 2), 1, 1, :))	# barotropic mode (nx, ny, nz + 2)

    u = copy(vars.u)                        # zonal velocity (nx, ny, nz + 2)
    v = copy(vars.v)                        # meridional velocity (nx, ny, nz + 2)
    selectdim(u, 3, 1) .= 0.5 .* (selectdim(vars.u, 3, 1) .+ selectdim(vars.u, 3, 2))                # averages using ghost point to give u_1/2
    selectdim(u, 3, nz + 2) .= 0.5 .* (selectdim(vars.u, 3, nz + 1) .+ selectdim(vars.u, 3, nz + 2)) # averages using ghost point to give u_N+1/2

    u₀ = mean(u .* ϕ₀, dims = 3)	        # barotropic zonal velocity (nx, ny, 1)
	v₀ = mean(v .* ϕ₀, dims = 3)	        # barotropic meridional velocity (nx, ny, 1)

	E₀ = mean(0.5 .* (u₀.^2 .+ v₀.^2))	    # domain integrated barotropic EKE (scalar)

    return E₀
end

function FirstBaroclinicEKE(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

	nz = params.nlevels				# number of interior levels
	H = collect(params.H)           # array of layer heights (nz)
	H₀ = sum(H)						# total depth [m]
	z = range(0., -H₀, nz + 1)      # vertical cell edges
	zc = z[1 : end - 1] .- H ./ 2   # vertical cell centres

	ϕ₁ = A(reshape(sqrt(2) * cos.(pi .* zc ./ H₀), 1, 1, :))	# first baroclinic mode (nx, ny, nz)

	u = view(vars.u, :, :, 2 : nz + 1)	# interior zonal velocity (nx, ny, nz)
	v = view(vars.v, :, :, 2 : nz + 1)	# interior meridional velocity (nx, ny, nz)

    u₁ = mean(u .* ϕ₁, dims = 3)		# first baroclinic zonal velocity (nx, ny, 1)
	v₁ = mean(v .* ϕ₁, dims = 3)		# first baroclinic meridional velocity (nx, ny, 1))

	E₁ = mean(0.5 .* (u₁.^2 .+ v₁.^2))	# domain integrated first baroclinic EKE (scalar)

    return E₁
end

function FullEKE(prob)
	# This function is organised slightly differently to the other diagnostic functions
	# so that variables are updated properly and I can print them as the model runs
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
	B = device_array(CPU())
    nz = params.nlevels				# number of interior levels

    @. vars.qh = sol
    MultiLevelQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

    @. vars.uh = -im * grid.l  * vars.ψh
    @. vars.vh =  im * grid.kr * vars.ψh

    MultiLevelQG.invtransform!(vars.u, vars.uh, params)
    MultiLevelQG.invtransform!(vars.v, vars.vh, params)

	u = copy(vars.u)                        # zonal velocity (nx, ny, nz + 2)
    selectdim(u, 3, 1) .= 0.5 .* (selectdim(vars.u, 3, 1) .+ selectdim(vars.u, 3, 2))                # averages using ghost point to give u_1/2
    selectdim(u, 3, nz + 2) .= 0.5 .* (selectdim(vars.u, 3, nz + 1) .+ selectdim(vars.u, 3, nz + 2)) # averages using ghost point to give u_N+1/2

    v = copy(vars.v)                        # meridional velocity (nx, ny, nz + 2)
    selectdim(v, 3, 1) .= 0.5 .* (selectdim(vars.v, 3, 1) .+ selectdim(vars.v, 3, 2))                # averages using ghost point to give u_1/2
    selectdim(v, 3, nz + 2) .= 0.5 .* (selectdim(vars.v, 3, nz + 1) .+ selectdim(vars.v, 3, nz + 2)) # averages using ghost point to give u_N+1/2

	E = dropdims(mean(0.5 .* (u.^2 + v.^2), dims = (1, 2)), dims = (1, 2))	# full EKE profile (nz + 2)

	return B(E)
end

function FirstBaroclinicDiffusivity(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
    B = device_array(CPU())

	nz = params.nlevels				# number of interior levels
	H = collect(params.H)           # array of layer heights (nz)
	H₀ = sum(H)						# total depth [m]
	z = range(0., -H₀, nz + 1)      # vertical cell edges
	zc = z[1 : end - 1] .- H ./ 2   # vertical cell centres

	ϕ₀ = A(reshape(ones(nz), 1, 1, :))							# barotropic mode (nx, ny, nz)
	ϕ₁ = A(reshape(sqrt(2) * cos.(pi .* zc ./ H₀), 1, 1, :))	# first baroclinic mode (nx, ny, nz)

	v = view(vars.v, :, :, 2 : nz + 1)		# interior meridional velocity (nx, ny, nz)
	ψ = view(vars.ψ, :, :, 2 : nz + 1)		# interior stream function (nx, ny, nz)

	v₀ = mean(v .* ϕ₀, dims = 3)	# barotropic meridional velocity (nx, ny, 1)
	ψ₁ = mean(ψ .* ϕ₁, dims = 3)	# first baroclinic streamfunction (nx, ny, 1)
    
    U = view(params.U, :, :, 2 : nz + 1)  # interior background velocity (nx, ny, nz)
	U₁ = mean(A(U) .* ϕ₁)	              # first baroclinic mean shear (scalar)

	D₁ = mean(v₀ .* ψ₁ ./ U₁)		# domain integrated first baroclinic eddy diffusivity (scalar)

	return D₁
end

function PVDiffusivity(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())
    nz = params.nlevels				# number of interior levels

	v = view(vars.v, :, :, 2 : nz + 1)	# interior meridional velocity (nx, ny, nz)
	q = view(vars.q, :, :, 2 : nz + 1)  # interior PV (nx, ny, nz)
	
	Qy = view(params.Qy, :, :, 2 : nz + 1)   # background meridional interior PV gradient (nx, ny, nz)
	β = params.β							 # background PV gradient from β (scalar)

	Qy_U = similar(Qy)
    copyto!(Qy_U, Qy)
    @views @. Qy_U[:, :, :] -= β	         # background PV gradient from thermal wind shear (nz)

	D = dropdims(mean((v .* q) ./ (-1 .* Qy), dims = (1, 2)), dims = (1, 2))	# PV diffusivity profile (nz)

	return B(D)
end

function FirstBaroclinicMixingLength(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

	nz = params.nlevels				# number of interior levels
	H = collect(params.H)           # array of layer heights (nz)
	H₀ = sum(H)						# total depth [m]
	z = range(0., -H₀, nz + 1)      # vertical cell edges
	zc = z[1 : end - 1] .- H ./ 2   # vertical cell centres

	ϕ₁ = A(reshape(sqrt(2) * cos.(pi .* zc ./ H₀), 1, 1, :))	# first baroclinic mode (nx, ny, nz)

	ψ = view(vars.ψ, :, :, 2 : nz + 1)    # interioer stream function (nx, ny, nz)
	ψ₁ = mean(ψ .* ϕ₁, dims = 3)		  # first baroclinic streamfunction (nx, ny, 1)

	U = view(params.U, :, :, 2 : nz + 1)  # interior background velocity (nx, ny, nz)
	U₁ = mean(A(U) .* ϕ₁)	              # first baroclinic mean shear (scalar)

	l₁ = sqrt(mean(ψ₁.^2 ./ U₁^2))		# domain integrated first baroclinic eddy mixing length (scalar)

	return l₁
end

function PVMixingLength(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())
    nz = params.nlevels				# number of interior levels

	q = view(vars.q, :, :, 2 : nz + 1)       # interior PV (nx, ny, nz)
	
	Qy = view(params.Qy, :, :, 2 : nz + 1)   # background meridional interior PV gradient (nx, ny, nz)
	β = params.β							 # background PV gradient from β (scalar)

	Qy_U = similar(Qy)
    copyto!(Qy_U, Qy)
    @views @. Qy_U[:, :, :] -= β	         # background PV gradient from thermal wind shear (nz)

	l = sqrt.(dropdims(mean((q.^2) ./ (Qy.^2), dims = (1, 2)), dims = (1, 2)))	# PV mixing length profile (nz)

	return B(l)
end

end
