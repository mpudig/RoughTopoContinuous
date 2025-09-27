# Some useful extra functions for running my experiments in GeophysicalFlows

module Utils

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler
using FourierFlows: parsevalsum

"""
	monoscale_random(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D topography field defined by a single length scale with random phases. 
"""

function monoscale_random(hrms, Ktopo, Lx, nx, dev)

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
	goff_jordan_iso(hrms, Ktopo, Lx, nx, dev, T)

Returns a 2D, isotropic topography field defined by the Goff Jordan spectrum with random phases. 
"""

function goff_jordan_iso(h_star, f0, U0, H0, Lx, nx, dev)

	 # Wavenumber grid
	 nk = Int(nx / 2 + 1)
	 nl = nx
	
	 dk = 2 * pi / Lx
	 dl = dk
	 
	 k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	 l = reshape( fftfreq(nx, dl * nx), (1, nl) )

	 K = @. sqrt(k^2 + l^2)

	 # Goff Jordan spectrum assuming isotropy
	 mu = 3.5
	 k0 = 1.8e-4
	 l0 = 1.8e-4

	 Random.seed!(1234)
	 hspec = @. 2 * pi * (mu - 2) / (k0 * l0) * (1 + (k / k0)^2 + (l / l0)^2)^(-mu / 2)
	 hh = hspec .* exp.(2 * pi * im .* rand(nk, nl))

	 # Recover h from hh
	 h = irfft(hh, nx)

	 # Get peak in topographic spectrum for calculating h_*
	 dhdx = irfft(im .* k .* hh, nx)
	 dhdy = irfft(im .* l .* hh, nx)
	 Ktopo = sqrt(mean(dhdx.^2 .+ dhdy.^2)) / sqrt(mean(h.^2))
	 
	 # Get hrms for given h_star, and scale h
	 hrms = h_star * U0 * H0 * Ktopo / f0
	 c = hrms / sqrt.(mean(h.^2))
	 h = c .* h

	 return h
end


"""
	set_initial_condition!(prob, grid, K0, E0)

	Sets the initial condition of MultiLayerQG to be a random q(x,y) field with baroclinic structure
	and with energy localized in spectral space about K = K0 and with total kinetic energy equal to E0
"""

function set_initial_condition!(prob, E0, K0, Kd)
	params = prob.params
	grid = prob.grid
	vars = prob.vars
	dev = grid.device
	T = eltype(grid)
	A = device_array(dev)

	# Grid
	nx = grid.nx
	Lx = grid.Lx

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

	psih = zeros(nk, nl, 2) .* im
	psih[:,:,1] = psihmag
	psih[:,:,2] = -1 .* psihmag

	# Calculate KE and APE, and prescribe mean total energy
	H = params.H
	V = grid.Lx * grid.Ly * sum(H)
	f0, gp = params.f₀, params.g′
	
	abs²∇𝐮h = zeros(nk, nl, 2) .* im
    abs²∇𝐮h[:,:,1] = K2 .* abs2.(psih[:,:,1])
    abs²∇𝐮h[:,:,2] = K2 .* abs2.(psih[:,:,2])

    KE = 1 / (2 * V) * (parsevalsum(abs²∇𝐮h[:,:,1], grid) * H[1] + parsevalsum(abs²∇𝐮h[:,:,1], grid) * H[2])
    APE = 1 / (2 * V) * f0^2 / gp * parsevalsum(abs2.(psih[:,:,1] .- psih[:,:,2]), grid)
    E = KE + APE
    c = sqrt(E0 / E)
    psih = @. c * psih

    # Invert psih to get qh, then transform qh to real space qh
    qh = zeros(nk, nl, 2) .* im
    qh[:,:,1] = - K2 .* psih[:,:,1] .+ f0^2 / (gp * H[1]) .* (psih[:,:,2] .- psih[:,:,1])
    qh[:,:,2] = - K2 .* psih[:,:,2] .+ f0^2 / (gp * H[2]) .* (psih[:,:,1] .- psih[:,:,2])

    q = zeros(nx, nx, 2)
    q[:,:,1] = irfft(qh[:,:,1], nx)
    q[:,:,2] = irfft(qh[:,:,2], nx)

    # Set as initial condition
    MultiLayerQG.set_q!(prob, A(q))
end

### Calculate diagnostics ###

function calc_KE(prob)
		vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
		nlayers = 2
		KE = zeros(nlayers)
			
		@. vars.qh = sol
		MultiLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
			
		abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
		@. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)
			
		V = grid.Lx * grid.Ly * sum(params.H)
			
		ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)
			
		for j = 1:nlayers
			  view(KE, j) .= 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * params.H[j]
		end
			  
		return KE
end
  
function calc_APE(prob)
		vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
			
		V = grid.Lx * grid.Ly * sum(params.H)

		ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)
			
		APE = 1 / (2 * V) * params.f₀^2 / params.g′ * parsevalsum(abs2.(ψ1h .- ψ2h), grid)
			  
		return APE
end

function calc_meridiff(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	psi1, psi2 = view(vars.ψ, :, :, 1), view(vars.ψ, :, :, 2)
	v1, v2 = view(vars.v, :, :, 1), view(vars.v, :, :, 2)

	psi_bc = 0.5 .* (psi1 - psi2)
	v_bt = 0.5 .* (v1 + v2)

	U1 = view(params.U, 1, 1, 1)
    U2 = view(params.U, 1, 1, 2)
    U0 = 0.5 * (U1 - U2)

	D = mean(psi_bc .* v_bt ./ U0)
		  
	return D
end

function calc_meribarovel(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	v1, v2 = view(vars.v, :, :, 1), view(vars.v, :, :, 2)
	v_bt = 0.5 .* (v1 + v2)

	V = sqrt(mean(v_bt.^2))
		  
	return V
end

function calc_mixlen(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	psi1, psi2 = view(vars.ψ, :, :, 1), view(vars.ψ, :, :, 2)
	psi_bc = 0.5 .* (psi1 - psi2)

	U1 = view(params.U, 1, 1, 1)
    U2 = view(params.U, 1, 1, 2)
    U0 = 0.5 * (U1 - U2)

	Lmix = sqrt(mean(psi_bc.^2) ./ U0.^2) 
	  
	return Lmix
end

function calc_KEFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	KEFlux1 = zeros(K_id)

	for j = 1:K_id
        # Get stream functions and vorticity
	    psih = vars.ψh
	    zetah = -grid.Krsq .* vars.ψh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih

		# Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)

		# Calculate spectral derivatives of zeta
		zetah_ik = im .* grid.kr .* zetah
		zetah_il = im .* grid.l .* zetah

		# Calculate real derivatives of zeta
		zeta_dx = A(zeros(nx, nx, nlayers))
		invtransform!(zeta_dx, zetah_ik, params)

		zeta_dy = A(zeros(nx, nx, nlayers))
		invtransform!(zeta_dy, zetah_il, params)

		# Create views of only upper layer fields
		psi_hpf_1 = view(psi_hpf, :, :, 1)
		u_1 = view(vars.u, :, :, 1)
		v_1 = view(vars.v, :, :, 1)
		zeta_dx_1 = view(zeta_dx, :, :, 1)
		zeta_dy_1 = view(zeta_dy, :, :, 1)

		view(KEFlux1, j) .= mean(psi_hpf_1 .* u_1 .* zeta_dx_1 + psi_hpf_1 .* v_1 .* zeta_dy_1)
	end
	return KEFlux1
end

function calc_APEFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	Ld = grid.Lx / (2 * pi * 25)

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	APEFlux1 = zeros(K_id)

    for j = 1:K_id
        # Get stream function
        psih = vars.ψh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih
        
        # Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)
        
        # Create views of single layer fields
        psi_hpf_1 = view(psi_hpf, :, :, 1)
        u1 = view(vars.u, :, :, 1)
        u2 = view(vars.u, :, :, 2)
        v1 = view(vars.v, :, :, 1)
        v2 = view(vars.v, :, :, 2)

		view(APEFlux1, j) .= mean(0.5 / Ld^2 .* psi_hpf_1 .* u1 .* v2 - 0.5 / Ld^2 .* psi_hpf_1 .* u2 .* v1)
	end
	return APEFlux1
end

function calc_ShearFlux_1(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	Ld = grid.Lx / (2 * pi * 25)
	U1 = view(params.U, 1, 1, 1)
    U2 = view(params.U, 1, 1, 2)
    U0 = 0.5 * (U1 - U2)

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)

	ShearFlux1 = zeros(K_id)

	for j = 1:K_id
        # Get stream function
	    psih = vars.ψh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih

		# Calculate spectral derivative
		psih_ik = im .* grid.kr .* psih

		# Inverse transforms
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)

		psi_dx = A(zeros(nx, nx, nlayers))
		invtransform!(psi_dx, psih_ik, params)

		# Views of necessary upper and lower layer fields
		psi_hpf_1 = view(psi_hpf, :, :, 1)
		psi_dx_2 = view(psi_dx, :, :, 2)

		# Calculate flux
		view(ShearFlux1, j) .= mean(U0 ./ Ld^2 .* psi_hpf_1 .* psi_dx_2)
	end
	return ShearFlux1
end

function calc_KEFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	KEFlux2 = zeros(K_id)

	for j = 1:K_id
        # Get stream functions and vorticity
	    psih = vars.ψh
	    zetah = -grid.Krsq .* vars.ψh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih

		# Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)

		# Calculate spectral derivatives of zeta
		zetah_ik = im .* grid.kr .* zetah
		zetah_il = im .* grid.l .* zetah

		# Calculate real derivatives of zeta
		zeta_dx = A(zeros(nx, nx, nlayers))
		invtransform!(zeta_dx, zetah_ik, params)

		zeta_dy = A(zeros(nx, nx, nlayers))
		invtransform!(zeta_dy, zetah_il, params)

		# Create views of only lower layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		u_2 = view(vars.u, :, :, 2)
		v_2 = view(vars.v, :, :, 2)
		zeta_dx_2 = view(zeta_dx, :, :, 2)
		zeta_dy_2 = view(zeta_dy, :, :, 2)

		view(KEFlux2, j) .= mean(psi_hpf_2 .* u_2 .* zeta_dx_2 + psi_hpf_2 .* v_2 .* zeta_dy_2)
	end
	return KEFlux2
end

function calc_APEFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	Ld = grid.Lx / (2 * pi * 25)

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	APEFlux2 = zeros(K_id)

    for j = 1:K_id
        # Get stream function
        psih = vars.ψh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih
        
        # Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)
        
        # Create views of single layer fields
        psi_hpf_2 = view(psi_hpf, :, :, 2)
        u1 = view(vars.u, :, :, 1)
        u2 = view(vars.u, :, :, 2)
        v1 = view(vars.v, :, :, 1)
        v2 = view(vars.v, :, :, 2)

		view(APEFlux2, j) .= mean(-0.5 / Ld^2 .* psi_hpf_2 .* u1 .* v2 + 0.5 / Ld^2 .* psi_hpf_2 .* u2 .* v1)
	end
	return APEFlux2
end

function calc_TopoFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	TopoFlux2 = zeros(K_id)

	for j = 1:K_id
        # Get stream function, velocity and topography
	    psih = vars.ψh
	    eta = A(params.eta)
        u = vars.u
        v = vars.v

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih

		# Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)

		# Calculate forward transforms of products 
		uetah = A(zeros(nk, nl, nlayers)) .* im
		fwdtransform!(uetah, u .* eta, params)

        vetah = A(zeros(nk, nl, nlayers)) .* im
		fwdtransform!(vetah, v .* eta, params)

        # Calculate spectral derivatives of products
		uetah_ik = im .* grid.kr .* uetah
		vetah_il = im .* grid.l .* vetah
		
		# Calculate real derivatives of products
		ueta_dx = A(zeros(nx, nx, nlayers))
		invtransform!(ueta_dx, etah_ik, params)

		veta_dy = A(zeros(nx, nx, nlayers))
		invtransform!(veta_dy, vetah_il, params)

		# Create views of only lower layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		ueta_dx_2 = view(ueta_dx, :, :, 2)
		veta_dy_2 = view(veta_dy, :, :, 2)

		view(TopoFlux2, j) .= mean(psi_hpf_2 .* ueta_dx_2 + psi_hpf_2 .* veta_dy_2)
	end
	return TopoFlux2
end

function calc_DragFlux_2(prob)
	vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol

	### Create isotropic wavenumber grid ###
	nk = Int(nx / 2 + 1)
	nl = nx

	kr = prob.grid.kr
	l = prob.grid.l
	Kr = @. sqrt(kr^2 + l^2)

	krmax = maximum(kr)
	lmax = maximum(abs.(l))
	Kmax = sqrt(krmax^2 + lmax^2)
	Kmin = 0.

	dkr = 2 * pi / Lx
	dl = dkr
	dKr = sqrt(dkr^2 + dl^2)
 
	K = Kmin:dKr:Kmax-dKr
	K_id = lastindex(K)
	
	DragFlux2 = zeros(K_id)

	for j = 1:K_id
        # Get velocities
	    uh = vars.uh
	    vh = vars.vh

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		uh_hpf = hpf .* uh
		vh_hpf = hpf .* vh

		# Inverse transform the filtered fields
		u_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(u_hpf, uh_hpf, params)

		v_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(v_hpf, vh_hpf, params)

		# Create views of only lower layer fields
		u_hpf_2 = view(u_hpf, :, :, 2)
		v_hpf_2 = view(v_hpf, :, :, 2)

		# Calculate drag flux
		view(DragFlux2, j) .= mean(-2 * μ .* u_hpf.^2 - 2 * μ .* v_hpf.^2)
	end
	return DragFlux2
end

end