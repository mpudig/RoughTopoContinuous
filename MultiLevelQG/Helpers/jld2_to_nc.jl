using NCDatasets, JLD2

# include and import parameters

include("../params.jl")
import .Params

# 3D fields
function convert_to_nc_fields()
    # Get path and open jld2 file

    expt_name = Params.expt_name
    file_path = Params.path_name[1:end-5] * "_fields" * ".jld2"
    file = jldopen(file_path)

    # Get necessary key information from file
    # Clock

    dt = file["clock/dt"]

    # Grid

    nx = Params.nx
    ny = nx
    nz = Params.nz
    Lx = Params.Lx
    Ly = Lx
    x = file["grid/x"]
    x = -x[1] .+ x
    y = file["grid/y"]
    y = -y[1] .+ y

    # Params

    f0 = Params.f₀
    beta = Params.β
    z = Params.z
    H0 = Params.H₀
    r = Params.r
    U0 = Params.U₀
    Ld = Params.Ld
    htop = Params.h
    U = Params.U
    N2 = Params.N²

    # Time and diagnostics

    iterations = parse.(Int, keys(file["snapshots/t"]))
    t = [file["snapshots/t/$iteration"] for iteration in iterations]

    # This creates a new NetCDF file
    # The mode "c" stands for creating a new file (clobber); the mode "a" stands for opening in write mode

    nc_path = file_path[1:end-5] * ".nc"
    if isfile(nc_path); rm(nc_path); end
    ds = NCDataset(nc_path, "c")
    ds = NCDataset(nc_path, "a")

    # Define attributes

    ds.attrib["title"] = expt_name
    ds.attrib["dt"] = dt
    ds.attrib["f0"] = f0
    ds.attrib["beta"] = beta
    ds.attrib["r"] = r
    ds.attrib["H"] = H0
    ds.attrib["Ld"] = Ld
    ds.attrib["U0"] = U0

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "z", size(z)[1])
    defDim(ds, "t", size(t)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "z", Float64, ("z",))
    ds["z"][:] = z

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:, :] = htop

    defVar(ds, "N2", Float64, ("z",))
    ds["N2"][:] = N2

    defVar(ds, "U", Float64, ("z",))
    ds["U"][:] = U

    defVar(ds, "q", Float64, ("x", "y", "z", "t"))
    for i in 1:length(iterations)
        iter = iterations[i]
        ds["q"][:, :, :, i] = file["snapshots/q/$iter"]
    end

    ### --- Save restart file comprising final snapshot of q --- ###
    restart_path = "../../output" * expt_name * "_restart.nc"
    if isfile(restart_path); rm(restart_path); end
    restart = NCDataset(restart_path, "c")
    restart = NCDataset(restart_path, "a")
    
    defDim(restart, "x", size(x)[1])
    defVar(restart, "x", Float64, ("x",))
    restart["x"][:] = x

    defDim(restart, "y", size(y)[1])
    defVar(restart, "y", Float64, ("y",))
    restart["y"][:] = y

    defDim(restart, "z", size(z)[1])
    defVar(restart, "z", Float64, ("z",))
    restart["z"][:] = z

    defVar(restart, "q", Float64, ("x", "y", "z"))
    restart["q"][:, :, :] = ds["q"][:, :, :, end]

    restart.attrib["t"] = t[end]

    # Finally, after all the work is done, we can close the files
    close(file)
    close(ds)
    close(restart)

    # Delete jld2 file
    rm(file_path)
end

# Diagnostics
function convert_to_nc_diags()
    # Get path and open jld2 file

    expt_name = Params.expt_name
    file_path = Params.path_name[1:end-5] * "_diags" * ".jld2"
    file = jldopen(file_path)

    # Get necessary key information from file
    # Clock

    dt = file["clock/dt"]

    # Grid

    nx = Params.nx
    ny = nx
    nz = Params.nz
    Lx = Params.Lx
    Ly = Lx
    x = file["grid/x"]
    x = -x[1] .+ x
    y = file["grid/y"]
    y = -y[1] .+ y

    # Params

    f0 = Params.f₀
    beta = Params.β
    z = Params.z
    H0 = Params.H₀
    r = Params.r
    U0 = Params.U₀
    Ld = Params.Ld
    htop = Params.h
    U = Params.U
    N2 = Params.N²

    # Time and diagnostics

    iterations = parse.(Int, keys(file["snapshots/t"]))
    t = [file["snapshots/t/$iteration"] for iteration in iterations]

    EKE = [file["snapshots/EKE/$iteration"] for iteration in iterations]
    EKE = reduce(hcat, EKE)

    vq = [file["snapshots/vq/$iteration"] for iteration in iterations]
    vq = reduce(hcat, vq)

    vb = [file["snapshots/vb/$iteration"] for iteration in iterations]
    vb = reduce(hcat, vb)

    qsq = [file["snapshots/qsq/$iteration"] for iteration in iterations]
    qsq = reduce(hcat, qsq)
    
    E0 = [file["snapshots/E₀/$iteration"] for iteration in iterations]
    E1 = [file["snapshots/E₁/$iteration"] for iteration in iterations]

    # This creates a new NetCDF file
    # The mode "c" stands for creating a new file (clobber); the mode "a" stands for opening in write mode

    nc_path = file_path[1:end-5] * ".nc"
    if isfile(nc_path); rm(nc_path); end
    ds = NCDataset(nc_path, "c")
    ds = NCDataset(nc_path, "a")

    # Define attributes

    ds.attrib["title"] = expt_name
    ds.attrib["dt"] = dt
    ds.attrib["f0"] = f0
    ds.attrib["beta"] = beta
    ds.attrib["r"] = r
    ds.attrib["H"] = H0
    ds.attrib["Ld"] = Ld
    ds.attrib["U0"] = U0

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "z", size(z)[1])
    defDim(ds, "t", size(t)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "z", Float64, ("z",))
    ds["z"][:] = z

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:, :] = htop

    defVar(ds, "N2", Float64, ("z",))
    ds["N2"][:] = N2

    defVar(ds, "U", Float64, ("z",))
    ds["U"][:] = U

    defVar(ds, "EKE", Float64, ("z", "t"))
    ds["EKE"][:, :] = EKE

    defVar(ds, "vq", Float64, ("z", "t"))
    ds["vq"][:, :] = vq

    defVar(ds, "vb", Float64, ("z", "t"))
    ds["vb"][:, :] = vb

    defVar(ds, "qsq", Float64, ("z", "t"))
    ds["qsq"][:, :] = qsq

    defVar(ds, "E0", Float64, ("t",))
    ds["E0"][:] = E0

    defVar(ds, "E1", Float64, ("t",))
    ds["E1"][:] = E1

    # Finally, after all the work is done, we can close the file and the dataset
    close(file)
    close(ds)

    # Delete jld2 file
    rm(file_path)
end
