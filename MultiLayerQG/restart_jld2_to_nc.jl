using NCDatasets, JLD2

# include and import parameters

include("params.jl")
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
    H = Params.H
    H0 = Params.H₀
    mu = Params.μ
    U0 = Params.U₀
    Ld = Params.Ld
    htop = Params.h
    U = Params.U
    b = Params.b

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
    ds.attrib["mu"] = mu
    ds.attrib["H"] = H0
    ds.attrib["Ld"] = Ld
    ds.attrib["U0"] = U0

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "lev", nz)
    defDim(ds, "t", size(t)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "lev", Int64, ("lev",))
    ds["lev"][:] = 1:1:nz

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:, :] = htop

    defVar(ds, "b", Float64, ("lev",))
    ds["b"][:] = b

    defVar(ds, "U", Float64, ("lev",))
    ds["U"][:] = U

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
    H = Params.H
    H0 = Params.H₀
    mu = Params.μ
    U0 = Params.U₀
    Ld = Params.Ld
    htop = Params.h
    U = Params.U
    b = Params.b

    # Time and diagnostics

    iterations = parse.(Int, keys(file["snapshots/t"]))
    t = [file["snapshots/t/$iteration"] for iteration in iterations]

    EKE = [file["snapshots/EKE/$iteration"] for iteration in iterations]
    EKE = reduce(hcat, EKE)

    D = [file["snapshots/D/$iteration"] for iteration in iterations]
    D = reduce(hcat, D)

    l = [file["snapshots/l/$iteration"] for iteration in iterations]
    l = reduce(hcat, l)
    
    E0 = [file["snapshots/E₀/$iteration"] for iteration in iterations]
    #E1 = [file["snapshots/E₁/$iteration"] for iteration in iterations]
    #D1 = [file["snapshots/D₁/$iteration"] for iteration in iterations]
    #l1 = [file["snapshots/l₁/$iteration"] for iteration in iterations]

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
    ds.attrib["mu"] = mu
    ds.attrib["H"] = H0
    ds.attrib["Ld"] = Ld
    ds.attrib["U0"] = U0

    # Define the dimensions, with names and sizes

    defDim(ds, "x", size(x)[1])
    defDim(ds, "y", size(y)[1])
    defDim(ds, "lev", nz)
    defDim(ds, "t", size(t)[1])

    # Define coordinates (i.e., variables with the same name as dimensions)

    defVar(ds, "x", Float64, ("x",))
    ds["x"][:] = x

    defVar(ds, "y", Float64, ("y",))
    ds["y"][:] = y

    defVar(ds, "lev", Int64, ("lev",))
    ds["lev"][:] = 1:1:nz

    defVar(ds, "t", Float64, ("t",))
    ds["t"][:] = t

    # Define variables: fields, diagnostics, snapshots

    defVar(ds, "htop", Float64, ("x", "y"))
    ds["htop"][:, :] = htop

    defVar(ds, "b", Float64, ("lev",))
    ds["b"][:] = b

    defVar(ds, "U", Float64, ("lev",))
    ds["U"][:] = U

    defVar(ds, "EKE", Float64, ("lev", "t"))
    ds["EKE"][:, :] = EKE

    defVar(ds, "E0", Float64, ("t",))
    ds["E0"][:] = E0

    #defVar(ds, "E1", Float64, ("t",))
    #ds["E1"][:] = E1

    defVar(ds, "D", Float64, ("lev", "t"))
    ds["D"][:, :] = D

    #defVar(ds, "D1", Float64, ("t",))
    #ds["D1"][:] = D1

    defVar(ds, "l", Float64, ("lev", "t"))
    ds["l"][:, :] = l

    #defVar(ds, "l1", Float64, ("t",))
    #ds["l1"][:] = l1

    # Finally, after all the work is done, we can close the file and the dataset
    close(file)
    close(ds)

    # Delete jld2 file
    rm(file_path)
end
