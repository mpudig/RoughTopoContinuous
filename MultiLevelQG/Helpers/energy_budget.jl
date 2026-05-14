# Energy budget diagnostics

module EnergyBudget

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, LinearAlgebra

"""
	KETransfer(prob)

Returns KE transfer in spectral space, integrated azimuthally.
"""
function KETransfer(prob)
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
    T_ke = A(zeros(T, nkr, nl, nz))          # KE spectral transfer (nkr, nl, nz)
    ζ = A(zeros(T, nx, nx, nz))              # real space vorticity (nx, nx, nz)
    ζh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vorticity (nkr, nl, nz)
    Fh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral scratch variable (nkr, nl, nz)

    # Vorticity
    @. ζh = -grid.Krsq * vars.ψh
    MultiLevelQG.invtransform!(ζ, ζh, params)

    # Zonal flux
    MultiLevelQG.fwdtransform!(Fh, vars.u .* ζ, params)
    @. T_ke .= real(conj(vars.ψh) * (im * grid.kr * Fh))

    # Meridional flux
    MultiLevelQG.fwdtransform!(Fh, vars.v .* ζ, params)
    @. T_ke += real(conj(vars.ψh) * (im * grid.l * Fh))

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    T_ke_iso = 2 * π .* K_bins .* Array(W * reshape(T_ke, nkr * nl, nz)) # (nK, nz)
    
    return T_ke_iso
end

"""
	PETransfer(prob)

Returns PE transfer in spectral space, integrated azimuthally.
"""
function PETransfer(prob)
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
    T_pe = A(zeros(T, nkr, nl, nz))          # PE spectral transfer (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))              # real space buoyancy (nx, nx, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space buoyancy (nkr, nl, nz)
    Fh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral scratch variable (nkr, nl, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    # Zonal flux
    MultiLevelQG.fwdtransform!(Fh, vars.u .* b, params)
    @. T_pe .= real(conj(bh) * (im * grid.kr * Fh))

    # Meridional flux
    MultiLevelQG.fwdtransform!(Fh, vars.v .* b, params)
    @. T_pe += real(conj(bh) * (im * grid.l * Fh))

    # Scale by -1/N²
    scale = A(collect(-1 ./ params.N²))
    T_pe .*= reshape(scale, 1, 1, nz)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    T_pe_iso = 2 * π .* K_bins .* Array(W * reshape(T_pe, nkr * nl, nz)) # (nK, nz)
    
    return T_pe_iso
end

"""
	InteriorPresFluxDiv(prob)

Returns pressure flux divergence term in spectral KE budget coming only from w_int, integrated azimuthally.
"""
function InteriorPresFluxDiv(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space buoyancy (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))                # real space buoyancy (nx, nx, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    ## Interior forcing for -H < z < 0
    # Scratch variables
    Q1 = similar(vars.q)
    Q2 = similar(vars.q)
    Qh = similar(vars.qh)

    # Zonal part of Qx
    MultiLevelQG.invtransform!(Q1, im * grid.kr .* vars.uh, params)          # ∂xu
    MultiLevelQG.invtransform!(Q2, im * grid.kr .* bh, params)               # ∂xb
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂xu ∂xb)
    @views rhsh[:, :, 2 : end - 1] .= -2 * im * grid.kr .* Qh[:, :, 2 : end - 1]

    # Zonal part of Qy
    MultiLevelQG.invtransform!(Q1, im * grid.l .* vars.uh, params)           # ∂yu
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂yu ∂xb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.l .* Qh[:, :, 2 : end - 1]

    # Meridional part of Qx
    MultiLevelQG.invtransform!(Q1, im * grid.kr .* vars.vh, params)          # ∂xv
    MultiLevelQG.invtransform!(Q2, im * grid.l  .* bh, params)               # ∂yb
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂xv ∂yb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.kr .* Qh[:, :, 2 : end - 1]

    # Meridional part of Qy
    MultiLevelQG.invtransform!(Q1, im * grid.l .* vars.vh, params)           # ∂yv
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂yv ∂yb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.l .* Qh[:, :, 2 : end - 1]

    # Mean flow part
    ∂zU = reshape(reshape(params.U, nl, nz) * params.D', 1, nl, nz)[:, :, 2 : end - 1]
    @views rhsh[:, :, 2 : end - 1] .+= -2 * params.f₀ * ∂zU .* grid.Krsq .* vars.ψh[:, :, 2 : end - 1]

    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle wh for InteriorPresFluxDiv, Re[-f₀ ∂z (ŵ* ψ̂)]
    D_wh_ψh = reshape(real(conj.(wh) .* vars.ψh), nkr * nl, nz)
    mul!(D_wh_ψh, copy(D_wh_ψh), params.D')
    wh .= -params.f₀ * reshape(D_wh_ψh, nkr, nl, nz)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    InteriorPresFluxDiv_iso = 2 * π .* K_bins .* Array(W * reshape(wh, nkr * nl, nz)) # (nK, nz)
    
    return InteriorPresFluxDiv_iso
end

"""
	DragPresFluxDiv(prob)

Returns pressure flux divergence term in spectral KE budget coming only from w_drag, integrated azimuthally.
"""
function DragPresFluxDiv(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    
    ## Lower BC: w = rζ at z = -H
    @views rhsh[:, :, end]  .= -params.r * grid.Krsq .* vars.ψh[:, :, end]

    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle wh for DragPresFluxDiv, Re[-f₀ ∂z (ŵ* ψ̂)]
    D_wh_ψh = reshape(real(conj.(wh) .* vars.ψh), nkr * nl, nz)
    mul!(D_wh_ψh, copy(D_wh_ψh), params.D')
    wh .= -params.f₀ * reshape(D_wh_ψh, nkr, nl, nz)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    DragPresFluxDiv_iso = 2 * π .* K_bins .* Array(W * reshape(wh, nkr * nl, nz)) # (nK, nz)
    
    return DragPresFluxDiv_iso
end

"""
	TopoPresFluxDiv(prob)

Returns pressure flux divergence term in spectral KE budget coming only from w_topo, integrated azimuthally.
"""
function TopoPresFluxDiv(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    
    ## Lower BC: w = J(ψ, h) at z = -H
    @views rhsh[:, :, end] .+= im * grid.kr .* rfft(vars.u[:, :, end] .* params.eta) .+
                               im * grid.l  .* rfft(vars.v[:, :, end] .* params.eta)
    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle wh for TopoPresFluxDiv, Re[-f₀ ∂z (ŵ* ψ̂)]
    D_wh_ψh = reshape(real(conj.(wh) .* vars.ψh), nkr * nl, nz)
    mul!(D_wh_ψh, copy(D_wh_ψh), params.D')
    wh .= -params.f₀ * reshape(D_wh_ψh, nkr, nl, nz)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    TopoPresFluxDiv_iso = 2 * π .* K_bins .* Array(W * reshape(wh, nkr * nl, nz)) # (nK, nz)
    
    return TopoPresFluxDiv_iso
end

"""
	PEForcing(prob)

Returns PE forcing extraction in spectral space, integrated azimuthally.
"""
function PEForcing(prob)
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
    PE_force = A(zeros(T, nkr, nl, nz))      # PE forcing (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))              # real space buoyancy (nx, nx, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space buoyancy (nkr, nl, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    # Compute
    ∂zU = reshape(reshape(params.U, nl, nz) * params.D', 1, nl, nz)
    @. PE_force .= ∂zU * real(conj(vars.vh) * bh)

    # Scale by f₀/N²
    scale = A(collect(params.f₀ ./ params.N²))
    PE_force .*= reshape(scale, 1, 1, nz)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    PE_force_iso = 2 * π .* K_bins .* Array(W * reshape(PE_force, nkr * nl, nz)) # (nK, nz)
    
    return PE_force_iso
end

"""
	InteriorVertBuoyFlux(prob)

Returns vertical buoyancy flux term in spectral KE budget coming only from w_int, integrated azimuthally.
"""
function InteriorVertBuoyFlux(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space buoyancy (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))                # real space buoyancy (nx, nx, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    ## Interior forcing for -H < z < 0
    # Scratch variables
    Q1 = similar(vars.q)
    Q2 = similar(vars.q)
    Qh = similar(vars.qh)

    # Zonal part of Qx
    MultiLevelQG.invtransform!(Q1, im * grid.kr .* vars.uh, params)          # ∂xu
    MultiLevelQG.invtransform!(Q2, im * grid.kr .* bh, params)               # ∂xb
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂xu ∂xb)
    @views rhsh[:, :, 2 : end - 1] .= -2 * im * grid.kr .* Qh[:, :, 2 : end - 1]

    # Zonal part of Qy
    MultiLevelQG.invtransform!(Q1, im * grid.l .* vars.uh, params)           # ∂yu
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂yu ∂xb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.l .* Qh[:, :, 2 : end - 1]

    # Meridional part of Qx
    MultiLevelQG.invtransform!(Q1, im * grid.kr .* vars.vh, params)          # ∂xv
    MultiLevelQG.invtransform!(Q2, im * grid.l  .* bh, params)               # ∂yb
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂xv ∂yb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.kr .* Qh[:, :, 2 : end - 1]

    # Meridional part of Qy
    MultiLevelQG.invtransform!(Q1, im * grid.l .* vars.vh, params)           # ∂yv
    MultiLevelQG.fwdtransform!(Qh, Q1 .* Q2, params)                         # \hat(∂yv ∂yb)
    @views rhsh[:, :, 2 : end - 1] .+= -2 * im * grid.l .* Qh[:, :, 2 : end - 1]

    # Mean flow part
    ∂zU = reshape(reshape(params.U, nl, nz) * params.D', 1, nl, nz)[:, :, 2 : end - 1]
    @views rhsh[:, :, 2 : end - 1] .+= -2 * params.f₀ * ∂zU .* grid.Krsq .* vars.ψh[:, :, 2 : end - 1]

    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle rhsh for InteriorVertBuoyFlux, Re[ŵ* b̂]
    @. rhsh .= real(conj.(wh) * bh)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    InteriorVertBuoyFlux_iso = 2 * π .* K_bins .* Array(W * reshape(rhsh, nkr * nl, nz)) # (nK, nz)
    
    return InteriorVertBuoyFlux_iso
end

"""
	DragVertBuoyFlux(prob)

Returns vertical buoyancy flux term in spectral KE budget coming only from w_drag, integrated azimuthally.
"""
function DragVertBuoyFlux(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space buoyancy (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))                # real space buoyancy (nx, nx, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    ## Lower BC: w = rζ at z = -H
    @views rhsh[:, :, end]  .= -params.r * grid.Krsq .* vars.ψh[:, :, end]

    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle rhsh for DragVertBuoyFlux, Re[ŵ* b̂]
    @. rhsh .= real(conj.(wh) * bh)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    DragVertBuoyFlux_iso = 2 * π .* K_bins .* Array(W * reshape(rhsh, nkr * nl, nz)) # (nK, nz)
    
    return DragVertBuoyFlux_iso
end

"""
	TopoVertBuoyFlux(prob)

Returns vertical buoyancy flux term in spectral KE budget coming only from w_topo, integrated azimuthally.
"""
function TopoVertBuoyFlux(prob)
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
    wh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space vertical velocity (nkr, nl, nz)
    rhsh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vertical velocity interior forcing + boundary conditions (nkr, nl, nz)
    bh = A(zeros(Complex{T}, nkr, nl, nz))     # spectral space buoyancy (nkr, nl, nz)
    b = A(zeros(T, nx, nx, nz))                # real space buoyancy (nx, nx, nz)

    # Buoyancy
    MultiLevelQG.bfromstreamfunction!(b, vars.ψ, params, grid)
    MultiLevelQG.fwdtransform!(bh, b, params)

    ## Lower BC: w = J(ψ, h) at z = -H
    @views rhsh[:, :, end] .+= im * grid.kr .* rfft(vars.u[:, :, end] .* params.eta) .+
                               im * grid.l  .* rfft(vars.v[:, :, end] .* params.eta)

    # Vertical velocity
    MultiLevelQG.omegaeqn!(wh, rhsh, params, grid)

    # Recycle rhsh for TopoVertBuoyFlux, Re[ŵ* b̂]
    @. rhsh .= real(conj.(wh) * bh)

    # Integrate azimuthally
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                    # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)   # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                       # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                       # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                                # normalise rows

    K_bins = Kmin .+ (0.5 : nK) .* dK                                    # bin centres (nK,)
    TopoVertBuoyFlux_iso = 2 * π .* K_bins .* Array(W * reshape(rhsh, nkr * nl, nz)) # (nK, nz)
    
    return TopoVertBuoyFlux_iso
end

end
