# Energy budget diagnostics

module EnergyBudget

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler

"""
	KETransfer(prob)

Returns KE transfer in spectral space, binned into isotropic wavenumber.
"""
function KETransfer(prob)
    sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
    MultiLevelQG.updatevars!(vars, params, grid, sol)

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
    F = A(zeros(T, nx, nx, nz))              # real scratch variable (nx, nx, nz)
    ζh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral space vorticity (nkr, nl, nz)
    Fh = A(zeros(Complex{T}, nkr, nl, nz))   # spectral scratch variable (nkr, nl, nz)

    # Vorticity
    @. ζh = -grid.Krsq * vars.ψh
    MultiLevelQG.invtransform!(ζ, ζh, params)

    # Zonal flux
    @. F = vars.u * ζ
    MultiLevelQG.fwdtransform!(Fh, F, params)
    @. T_ke = real(conj(vars.ψh) * (im * grid.kr * Fh))

    # Meridional flux
    @. F = vars.v * ζ
    MultiLevelQG.fwdtransform!(Fh, F, params)
    @. T_ke += real(conj(vars.ψh) * (im * grid.l * Fh))

    # Average in rings
    kr = grid.kr
    l = grid.l
    Kr = @. sqrt(kr^2 + l^2)
    Kmin = 0.
    Kmax = sqrt(maximum(kr)^2 + maximum(abs.(l))^2)
    dK = 2 * pi / Lx
    nK = floor(Int, (Kmax - Kmin) / dK)

    Kr_flat = vec(Kr)                                                   # (nkr * nl,)
    iK_flat = clamp.(floor.(Int, (Kr_flat .- Kmin) ./ dK) .+ 1, 1, nK)  # (nkr * nl,)
    W = T.(iK_flat' .== A(1 : nK))                                      # (nK, nkr * nl)
    counts = vec(sum(W, dims = 2))                                      # (nK,)
    W ./= max.(reshape(counts, nK, 1), 1)                               # normalise rows

    T_ke_iso = Array(W * reshape(T_ke, nkr * nl, nz))                   # (nK, nz)
    
    return T_ke_iso
end


end
