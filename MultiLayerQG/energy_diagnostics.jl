"""
Goal: take the last snapshot for PV from a run,
initialize a model instance with this state,
use my energy budget diagnostics to calculate
the energy budget terms given this snapshot,
then save these terms in a .nc file.

I am doing this because for some reason 
the budget terms are not saving as model
diagnostics when I run the model forward.
Ultimately, I'd like to fix this bug but
at present I cannot figure out how to.
"""

using GeophysicalFlows, FFTW, Statistics, Random, Printf, JLD2, NCDatasets, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays, NCDatasets;

# local import
import .Utils
import .Params

# include and import parameters

include("params.jl")
import .Params

function calc_and_save_energy_budget()

    # Get path, open nc file, get final snapshot of q

    expt_name = Params.expt_name         
    path = "../../output" * expt_name * ".nc"
    ds = NCDataset(path, "r")
    qi = ds["q"][:, :, :, end]
    close(ds)

    # Create model instance using this qi and also parameters from params

     ### Grid ###

    nx = Params.nx
    Lx = Params.Lx

    nlayers = Params.nz
    H = Params.H

    dev = GPU()

    ### Planetary parameters ###

    f₀ = Params.f0
    β = Params.beta
    g = Params.g
    μ = Params.kappa
    Ld = Params.Ld
    U0 = Params.U0

    ρ = Params.rho
    U = Params.U
    eta = Params.eta

    ### Create model instance ###

    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, μ, eta)
    vars, params, grid, sol = prob.vars, prob.params, prob.grid, prob.sol
    A = device_array(grid.device)
    MultiLayerQG.set_q!(prob, A(qi))

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

    ##### ENERGY BUDGET TERMS BELOW ###

    ### KEFlux1 ###

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

    ### APEFlux1 ###

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
	
	### ShearFlux1 ###

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
		view(ShearFlux1, j) .= mean(U0 / Ld^2 .* psi_hpf_1 .* psi_dx_2)
	end

    ### KEFlux2 ###

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

    ### APEFlux2 ###

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

    ### TopoFlux2 ###

    TopoFlux2 = zeros(K_id)

	for j = 1:K_id
        # Get stream functions and topography
	    psih = vars.ψh
	    eta = A(params.eta)
        etah = A(zeros(nk, nl, nlayers)) .* im
        fwdtransform!(etah, eta, params)

		# Define high-pass filter matrix
		hpf = ifelse.(Kr .> K[j], Kr ./ Kr, 0 .* Kr)

		# Filter the Fourier transformed fields
		psih_hpf = hpf .* psih

		# Inverse transform the filtered fields
		psi_hpf = A(zeros(nx, nx, nlayers))
		invtransform!(psi_hpf, psih_hpf, params)

		# Calculate spectral derivatives of topography
		etah_ik = im .* grid.kr .* etah
		etah_il = im .* grid.l .* etah
		
		# Calculate real derivatives of topography
		eta_dx = A(zeros(nx, nx, nlayers))
		invtransform!(eta_dx, etah_ik, params)

		eta_dy = A(zeros(nx, nx, nlayers))
		invtransform!(eta_dy, etah_il, params)

		# Create views of only lower layer fields
		psi_hpf_2 = view(psi_hpf, :, :, 2)
		u_2 = view(vars.u, :, :, 2)
		v_2 = view(vars.v, :, :, 2)

		view(TopoFlux2, j) .= mean(psi_hpf_2 .* u_2 .* eta_dx + psi_hpf_2 .* v_2 .* eta_dy)
	end

    ### DragFlux2 ###

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

    ##### Here, I save the energy budget terms in a new .nc file #####

    path = "../../output" * expt_name * "energy_budget.nc"
    ds = NCDataset(path, "c")
    ds = NCDataset(path, "a")

    # Define attributes

    ds.attrib["title"] = expt_name

    # Define the dimensions, with names and sizes

    defDim(ds, "K", size(K)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "K", Float64, ("K",))
    ds["K"][:] = K

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "KEFlux1", Float64, ("K",))
    ds["KEFlux1"][:] = KEFlux1
    
    defVar(ds, "APEFlux1", Float64, ("K",))
    ds["APEFlux1"][:] = APEFlux1

    defVar(ds, "ShearFlux1", Float64, ("K",))
    ds["ShearFlux1"][:] = ShearFlux1

    defVar(ds, "KEFlux2", Float64, ("K",))
    ds["KEFlux2"][:] = KEFlux2

    defVar(ds, "APEFlux2", Float64, ("K",))
    ds["APEFlux2"][:] = APEFlux2

    defVar(ds, "TopoFlux2", Float64, ("K",))
    ds["TopoFlux2"][:] = TopoFlux2

    defVar(ds, "DragFlux2", Float64, ("K",))
    ds["DragFlux2"][:] = DragFlux2

    # Finally, after all the work is done, we can close the dataset
    close(ds)

end