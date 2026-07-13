# Some useful extra functions for running my experiments with GeophysicalFlows.jl

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, SpecialFunctions, ForwardDiff, LinearAlgebra
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
        set_initial_condition!(prob, grid, K0, E0, ϕₘ)

        Sets the initial condition of QG3D problem to be a random q(x,y) field with baroclinic structure ϕₘ
        and with energy localized in spectral space about K = K₀ and with total energy equal to E₀
"""
function set_initial_condition!(prob, K0, E0, ϕₘ)
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
    psih = zeros(nk, nl, nz) .* im
    for j = 2 : nz - 1
        psih[:, :, j] = psihmag .* ϕₘ[j]
    end
    CUDA.@allowscalar psih[:, :, 1] .= psih[:, :, 2]           # b|z=0 = 0 to first order (set exactly to zero later)
    CUDA.@allowscalar psih[:, :, end] .= psih[:, :, end - 1]   # b|z=-H = 0 to first order (set exactly to zero later)

    # Calculate KE and APE, and prescribe mean total energy
    # For simplicity, neglect the z = 0 and z = -H surfaces in the calculation
    z = params.z
    V = grid.Lx * grid.Ly * abs(z[end])
    f0, N² = params.f₀, params.N²

    abs²∇𝐮h = zeros(nk, nl, nz) .* im
    for j = 1 : nz
        abs²∇𝐮h[:, :, j] = K2 .* abs2.(psih[:, :, j])
    end

    KE, PE = zeros(nz), zeros(nz - 1)

    dz = -diff(collect(z))  
    for j = 2 : nz - 1
        KE[j] = 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * dz[j - 1]
    end

	for j = 1 : nz - 1
        PE[j] = 1 / (2 * V) * params.f₀^2 ./ params.N²[j] .* parsevalsum(abs2.(view(psih, :, :, j) .- view(psih, :, :, j + 1)), grid)
    end

    E = sum(KE) + sum(PE)
    c = sqrt(E0 / E)
    psih = @. c * psih
    psih = A(psih)

    # Invert psih to get qh, then transform qh to real space qh
    qh = A(zeros(nk, nl, nz)) .* im
    QG3D.pvfromstreamfunction!(qh, psih, params, grid)

    q = A(zeros(nx, nx, nz))
    QG3D.invtransform!(q, qh, params)
    CUDA.@allowscalar q[:, :, 1] .= 0           # b|z=0 = 0 exactly
    CUDA.@allowscalar q[:, :, end] .= 0         # b|z=-H = 0 exactly

    # Set as initial condition
    QG3D.set_q!(prob, q)
end

"""
        LinStratN(f₀, H, Ld)

Returns the N₀ for a given first baroclinic deformation radius with constant stratification
"""

function LinStratN(f₀, H₀, Ld)
	N₀ = pi * f₀ * Ld / H₀

	return N₀
end

"""
		ExpStratEigval1(δ)
Returns the first baroclinic eigenvalue for exponential stratification and given δ.
Uses a Newton method to compute it numerically (I know roughly where it'll be, hence the hard-coded first guess!)
"""

function ExpStratEigval1(δ)

	f(x) = besselj(0, x) * bessely(0, exp(-1/(2*δ)) * x) - bessely(0, x) * besselj(0, exp(-1/(2*δ)) * x)	# Equation needed to solve for eigenvalue (e.g., see LaCasce 2012)
	autodiff(f) = x -> ForwardDiff.derivative(f, x)		# Derivative of f (easier to do this numerically)

	function Newton(f, x0, tol = 1e-12, maxIter = 1e3)
		x = x0
		fx = f(x0)
		fp = autodiff(f)
		iter = 0
		while abs(fx) > tol && iter < maxIter
               x = x  - fx/fp(x)   # Iteration
               fx = f(x)           # Precompute f(x)
               iter += 1
           end
           return x
	end

	x0 = 2.					# Hard-coded initial guess
	a₁ = Newton(f, x0)		# Eigenvalue

	return a₁
end

"""
        ExpStratN(f₀, H, Ld)

Returns the N₀ for a given first baroclinic deformation radius with exponential stratification
"""

function ExpStratN(f₀, H₀, Ld, δ, a₁)
		N₀ = f₀ * a₁ * Ld / (2 * H₀ * δ)

	return N₀
end

"""
		ExpStratPhi1(z, δ, H₀, a₁)
Returns the first baroclinic eigenfunction for given exponential stratification on Chebyshev levels and normalized with unit depth-average.
"""

function ExpStratPhi1(z, δ, H₀, a₁)
	ϕ₁(z) = exp.(z ./ (2 * δ * H₀)) .* (bessely(0, a₁) .* besselj.(1, a₁ .* exp.(z ./ (2 * δ * H₀))) .- besselj(0, a₁) .* bessely.(1, a₁ .* exp.(z ./ (2 * δ * H₀))))

    function clencurt_weights(nz)
        n = nz - 1
        c = zeros(nz)
        c[1 : 2 : end] .= 2.0 ./ (1 .- (0 : 2 : n).^2)
        w = real(ifft([c; c[end - 1 : -1 : 2]])[1 : nz])
        w[1] /= 2
        w[end] /= 2
        return w
    end
    weights = clencurt_weights(length(z))
    norm = sqrt(sum(weights .* ϕ₁(z).^2))

	return ϕ₁(z) / norm
end

"""
        first_baroclinic_mode(f₀, H₀, N², nz)
Returns the first baroclinic mode, given N², f₀, H₀
"""
function first_baroclinic_mode(f₀, H₀, N², nz)
    # Chebyshev grid and differentiation matrix
    ζ = [cos(i * π / (nz - 1)) for i in 0:nz-1]
    c = ones(nz); c[1] = 2; c[end] = 2
    D = zeros(nz, nz)
    for i in 1:nz, j in 1:nz
        if i != j
            D[i, j] = (c[i] / c[j]) * (-1.0)^(i + j) / (ζ[i] - ζ[j])
        end
    end
    for i in 1:nz
        D[i, i] = -sum(D[i, j] for j in 1:nz if j != i)
    end
    D .*= 2 / H₀

    # Stretching operator: L = D * diag(f0^2/N2) * D
    L = D * Diagonal(f₀^2 ./ N²) * D

    # Neumann BC bordering (d/dz phi = 0 at z=0, z=-H)
    L_bc = copy(L)
    L_bc[1, :]   = D[1, :]
    L_bc[end, :] = D[end, :]
    B = Matrix{Float64}(I, nz, nz)
    B[1, :]   .= 0.0
    B[end, :] .= 0.0

    # Generalized eigenproblem: L_bc*phi = -lambda^2 * B*phi
    F = eigen(L_bc, B)
    vals, vecs = F.values, F.vectors

    finite0 = isfinite.(vals)
    tol = 1e-8 * maximum(abs.(vals[finite0]))
    mask = finite0 .& (real.(vals) .< tol)
    lambda2 = -real.(vals[mask])
    modes   =  real.(vecs[:, mask])

    idx = sortperm(lambda2)
    lambda2 = lambda2[idx]
    modes   = modes[:, idx]

    # Clenshaw-Curtis quadrature weights on [-H0, 0]
    cc = zeros(nz)
    cc[1:2:end] = 2.0 ./ (1 .- (0:2:nz-1).^2)
    w = real.(ifft(vcat(cc, cc[end-1:-1:2])))[1:nz]
    w[1]   /= 2
    w[end] /= 2
    weights = w * H₀

    inner(u, v) = dot(u, weights .* v) / H₀

    # Only need modes 1 (barotropic) and 2 (first baroclinic) for Gram-Schmidt
    for m in 1:2
        mode = modes[:, m]
        for j in 1:(m - 1)
            mode .-= inner(mode, modes[:, j]) .* modes[:, j]
        end
        mode ./= sqrt(inner(mode, mode))
        if mode[1] < 0
            mode .*= -1
        end
        modes[:, m] = mode
    end

    ϕ₁ = modes[:, 2]
    
    return ϕ₁
end


### Diagnostics ###

# Note: Clenshaw-Curtis quadrature weights are used to average in the vertical
# Note: The modal projection diagnostics assume constant stratification; they must be modified when using nonconstant stratification

### --- Vertical profiles --- ###
function FullEKE(prob)
	# This function is organised slightly differently to the other diagnostic functions
	# so that variables are updated properly and I can print them as the model runs
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
	B = device_array(CPU())
    nz = params.nlevels			   # number of levels

    @. vars.qh = sol
    QG3D.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

    @. vars.uh = -im * grid.l  * vars.ψh
    @. vars.vh =  im * grid.kr * vars.ψh

    QG3D.invtransform!(vars.u, vars.uh, params)
    QG3D.invtransform!(vars.v, vars.vh, params)

	u = view(vars.u, :, :, :)	   # interior zonal velocity (nx, ny, nz)
	v = view(vars.v, :, :, :)	   # interior meridional velocity (nx, ny, nz)

	E = dropdims(mean(0.5 .* (u.^2 + v.^2), dims = (1, 2)), dims = (1, 2))	# full EKE profile (nz)

	return B(E)
end

function PVFluxMeridional(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	B = device_array(CPU())

	vq = dropdims(mean(vars.v .* vars.q, dims = (1, 2)), dims = (1, 2))	# meridional PV flux profile (nz)

	return B(vq)
end

function BFluxMeridional(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
    B = device_array(CPU())

	b = similar(vars.q)
	QG3D.bfromstreamfunction!(b, vars.ψ, params, grid)
    vb = dropdims(mean(vars.v .* b, dims = (1, 2)), dims = (1, 2))     # meridional buoyancy flux profile (nz)

    return B(vb)
end

function PVFluxZonal(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	B = device_array(CPU())

	uq = dropdims(mean(vars.u .* vars.q, dims = (1, 2)), dims = (1, 2))	# zonal PV flux profile (nz)

	return B(uq)
end

function BFluxZonal(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
    B = device_array(CPU())

	b = similar(vars.q)
	QG3D.bfromstreamfunction!(b, vars.ψ, params, grid)
    ub = dropdims(mean(vars.u .* b, dims = (1, 2)), dims = (1, 2))     # zonal buoyancy flux profile (nz)

    return B(ub)
end

function PVVariance(prob)
	vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

	qsq = dropdims(mean(vars.q.^2, dims = (1, 2)), dims = (1, 2))	# PV variance profile (nz)

	return B(qsq)
end

### --- Scalars --- ###
function BarotropicEKE(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

    # Vertical grid
    nz = params.nlevels				# number of levels

    # Clenshaw-Curtis quadrature weights for vertical averaging on Chebyshev gird
    function clencurt_weights(nz)
        n = nz - 1
        c = zeros(nz)
        c[1 : 2 : end] .= 2.0 ./ (1 .- (0 : 2 : n).^2)
        w = real(ifft([c; c[end - 1 : -1 : 2]])[1 : nz])
        w[1] /= 2
        w[end] /= 2
        return w
    end
    weights = A(clencurt_weights(nz))
    #weights = A([(1.0 - sum(cos(2 * pi / n * k * i) / (4 * k^2 - 1) for k in 1:div(n, 2))) / n / (i == 0 || i == n ? 2 : 1) for i = 0:n]) # divide by 2 if i = 0 or i = n
    
    # Vertically average
    u₀ = sum(vars.u .* reshape(weights, 1, 1, :), dims = 3)	# barotropic zonal velocity (nx, ny, 1)
	v₀ = sum(vars.v .* reshape(weights, 1, 1, :), dims = 3)	# barotropic meridional velocity (nx, ny, 1)
	E₀ = mean(0.5 .* (u₀.^2 .+ v₀.^2))	                    # domain integrated barotropic EKE (scalar)

    return E₀
end

function FirstBaroclinicEKE(prob)
    vars, params, grid = prob.vars, prob.params, prob.grid
	A = device_array(grid.device)
	B = device_array(CPU())

    # Vertical grid
    nz = params.nlevels				# number of levels
	z = collect(params.z)           # vertical grid
	H₀ = params.H₀   				# total depth [m]

    # Clenshaw-Curtis-Clencurt quadrature weights for vertical averaging on Chebyshev gird
    n = nz - 1
    function clencurt_weights(nz)
        n = nz - 1
        c = zeros(nz)
        c[1 : 2 : end] .= 2.0 ./ (1 .- (0 : 2 : n).^2)
        w = real(ifft([c; c[end - 1 : -1 : 2]])[1 : nz])
        w[1] /= 2
        w[end] /= 2
        return w
    end
    weights = A(clencurt_weights(nz))
    #weights = A([(1.0 - sum(cos(2 * pi / n * k * i) / (4 * k^2 - 1) for k in 1:div(n, 2))) / n / (i == 0 || i == n ? 2 : 1) for i = 0:n]) # divide by 2 if i = 0 or i = n
    
    # Vertically average
    ϕ₁ = A(reshape(sqrt(2) * cos.(pi .* z ./ H₀), 1, 1, :))	        # first baroclinic mode (nx, ny, nz)
    u₁ = sum(vars.u .* ϕ₁ .* reshape(weights, 1, 1, :), dims = 3)	# first baroclinic zonal velocity (nx, ny, 1)
	v₁ = sum(vars.v .* ϕ₁ .* reshape(weights, 1, 1, :), dims = 3)	# first baroclinic meridional velocity (nx, ny, 1)
	E₁ = mean(0.5 .* (u₁.^2 .+ v₁.^2))	                            # domain integrated first baroclinic EKE (scalar)

    return E₁
end

"""
	w_topo(prob)

Returns real space w_topo.
"""
function w_topo(prob)
    sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

    # Grid
    A = device_array(grid.device)
    T = eltype(grid)
    Lx = grid.Lx
    nx = grid.nx
    nkr = grid.nkr
    nl = grid.nl
    nz = params.nlevels

    ### Calculate
    # Assign
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))  # spectral space omega equation rhs + boundary conditions (nkr, nl, nz)
    wh = A(zeros(Complex{T}, nkr, nl, nz))    # spectral space vertical velocity (nkr, nl, nz)
    w = A(zeros(T, nx, nx, nz))               # real space vertical velocity (nx, ny, nz)
    
    # Lower BC: w = J(ψ, h) at z = -H
    @views rhsh[:, :, end] .+= im * grid.kr .* rfft(vars.u[:, :, end] .* params.eta) .+
                               im * grid.l  .* rfft(vars.v[:, :, end] .* params.eta)
    # Vertical velocity
    QG3D.omegaeqn!(wh, rhsh, params, grid)
    QG3D.invtransform!(w, wh, params)
    
    return w
end

end
