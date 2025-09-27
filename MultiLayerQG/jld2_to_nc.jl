using NCDatasets, JLD2

# include and import parameters

include("params.jl")
import .Params

function convert_to_nc()
    # Get path and open jld2 file

    expt_name = Params.expt_name         
    file_path = "../../output" * expt_name * ".jld2"
    file = jldopen(file_path)

    # Get necessary key information from file
    # Clock

    dt = file["clock/dt"]

    # Grid

    nx = file["grid/nx"]
    ny = file["grid/ny"]
    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    x = file["grid/x"]
    x = -x[1] .+ x
    y = file["grid/y"]
    y = -y[1] .+ y

    # Params

    f0 = file["params/f₀"]
    beta = file["params/β"]
    rho = file["params/ρ"]
    rho1 = rho[1]
    rho2 = rho[2]
    U = file["params/U"][1,1,:]
    U1 = U[1]
    U2 = U[2]
    H = file["params/H"]
    delta = H[1] / H[2]
    H0 = sum(H)
    kappa = file["params/μ"]
    gp = file["params/g′"]
    eta = file["params/eta"]
    htop = H[1] / f0 .* eta
    Qx = file["params/Qx"]
    Qy = file["params/Qy"]

    # Time and diagnostics
    iterations = parse.(Int, keys(file["snapshots/t"]))
    t = [file["snapshots/t/$iteration"] for iteration in iterations]

    KE = [file["snapshots/KE/$iteration"] for iteration in iterations]
    KE = reduce(hcat, KE)
    APE = [file["snapshots/APE/$iteration"] for iteration in iterations]
    D = [file["snapshots/D/$iteration"] for iteration in iterations]
    V = [file["snapshots/V/$iteration"] for iteration in iterations]
    Lmix = [file["snapshots/Lmix/$iteration"] for iteration in iterations]

    #KEFlux1 = [file["snapshots/KEFlux1/$iteration"] for iteration in iterations]
    #APEFlux1 = [file["snapshots/APEFlux1/$iteration"] for iteration in iterations]
    #ShearFlux1 = [file["snapshots/ShearFlux1/$iteration"] for iteration in iterations]
    #KEFlux2 = [file["snapshots/KEFlux2/$iteration"] for iteration in iterations]
    #APEFlux2 = [file["snapshots/APEFlux2/$iteration"] for iteration in iterations]
    #TopoFlux2 = [file["snapshots/TopoFlux2/$iteration"] for iteration in iterations]
    #DragFlux2 = [file["snapshots/DragFlux2/$iteration"] for iteration in iterations]

    # Make isotropic wavenumber grid
    nx = file["grid/nx"]
    Lx = file["grid/Lx"]
    nk = Int(nx / 2 + 1)
	nl = nx
	dk = 2 * pi / Lx
	dl = dk
    k = reshape( rfftfreq(nx, dk * nx), (nk, 1) )
	l = reshape( fftfreq(nx, dl * nx), (1, nl) )
    kmax = maximum(k)
	lmax = maximum(abs.(l))
	Kmax = sqrt(kmax^2 + lmax^2)
	Kmin = 0.
	dK = sqrt(dk^2 + dl^2)
    K = Kmin:dK:Kmax-dKr

    ### From here, create a new NetCDF file ###
    # The mode "c" stands for creating a new file (clobber); the mode "a" stands for opening in write mode

    file_path_nc = "../../output" * expt_name * ".nc"
    #file_path_nc = "../../output" * expt_name * "_restart1" * ".nc"
    ds = NCDataset(file_path_nc, "c")
    ds = NCDataset(file_path_nc, "a")

    # Define attributes

    ds.attrib["title"] = expt_name
    ds.attrib["dt"] = dt
    ds.attrib["f0"] = f0
    ds.attrib["beta"] = beta
    ds.attrib["kappa"] = kappa
    ds.attrib["rho1"] = rho1
    ds.attrib["rho2"] = rho2
    ds.attrib["gp"] = gp
    ds.attrib["U1"] = U1
    ds.attrib["U2"] = U2
    ds.attrib["H"] = H0
    ds.attrib["delta"] = delta

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "lev", 2)
    defDim(ds, "t", size(t)[1])
    defDim(ds, "K", size(K)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "lev", Int64, ("lev",))
    ds["lev"][:] = [1, 2]

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # defVar(ds, "K", Float64, ("K",))
    # ds["K"][:] = K

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:,:] = htop

    defVar(ds, "KE", Float64, ("lev", "t"))
    ds["KE"][:,:] = KE

    defVar(ds, "APE", Float64, ("t",))
    ds["APE"][:] = APE

    defVar(ds, "D", Float64, ("t",))
    ds["D"][:] = D

    defVar(ds, "V", Float64, ("t",))
    ds["V"][:] = V

    defVar(ds, "Lmix", Float64, ("t",))
    ds["Lmix"][:] = Lmix

    # defVar(ds, "KEFlux1", Float64, ("K", "t"))
    # ds["KEFlux1"][:] = KEFlux1
    
    # defVar(ds, "APEFlux1", Float64, ("K", "t"))
    # ds["APEFlux1"][:] = APEFlux1

    # defVar(ds, "ShearFlux1", Float64, ("K", "t"))
    # ds["ShearFlux1"][:] = ShearFlux1

    # defVar(ds, "KEFlux2", Float64, ("K", "t"))
    # ds["KEFlux2"][:] = KEFlux2

    # defVar(ds, "APEFlux2", Float64, ("K", "t"))
    # ds["APEFlux2"][:] = APEFlux2

    # defVar(ds, "TopoFlux2", Float64, ("K", "t"))
    # ds["TopoFlux2"][:] = TopoFlux2

    # defVar(ds, "DragFlux2", Float64, ("K", "t"))
    # ds["DragFlux2"][:] = DragFlux2

    defVar(ds, "q", Float64, ("x", "y", "lev", "t"))
    for i in 1:length(iterations)
        iter = iterations[i]
        ds["q"][:,:,:,i] = file["snapshots/q/$iter"]
    end

    # Finally, after all the work is done, we can close the file and the dataset
    close(file)
    close(ds)

    # Delete jld2 file
    rm(file_path)
end