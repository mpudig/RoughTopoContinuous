# This is the driver: set up, run, and save the model
# This module assumes a .nc file exists to restart from
# and outputs new files given the name of files already in the folder

# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, Printf, JLD2, NCDatasets, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays, KernelAbstractions;
using FourierFlows: CPU, GPU
using Dates

# local import
import .Utils
import .Params

      ### Save path and device ###

path_name = Params.path_name
dev = Params.dev

      ### Grid ###

nx = Params.nx
nlayers = Params.nz

Lx = Params.Lx
H = Params.H

      ### Background parameters ###

f₀ = Params.f₀
β = Params.β
g = Params.g
μ = Params.μ

b = Params.b
U = Params.U

eta = Params.eta

      ### Time stepping ###

dt = Params.dt
tmax = Params.tmax
dtsnap_diags = Params.dtsnap_diags
dtsnap_fields = Params.dtsnap_fields
nsubs_diags = Params.nsubs_diags
nsubs_fields = Params.nsubs_fields
nsteps = Params.nsteps
stepper = Params.stepper

      ### Step the model forward ###

function simulate!(prob, grid, diags, EKE, out_fields, out_diags, tmax, nsteps, dtsnap_diags, dtsnap_fields, nsubs_diags, nsubs_fields)
      # Save problem
      saveproblem(out_fields)
      saveproblem(out_diags)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs_diags)

      for j = frames
      cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            if j % 5 == 0

                  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, KE1: %.3e, KE%d: %.3e, walltime: %.2f min",
                  clock.step, clock.t, cfl, EKE.data[EKE.i][1], nlayers, EKE.data[EKE.i][end], (time() - startwalltime) / 60)

                  println(log)
                  flush(stdout)
            end

            # If cfl is close to unstable value, halve the time step and reset the problem with initial condition at last time step
            if cfl > 0.85

                  # Reset time stepping variables
                  dt = clock.dt / 2
                  clock.dt = dt
                  nsubs_diags = Int(floor(dtsnap_diags / dt))
                  nsubs_fields = Int(floor(dtsnap_fields / dt))
                  nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs_fields) * nsubs_fields

                  # Reset diagnostics for new nsteps
                  # Energies
                  E₀ = Diagnostic(Utils.BarotropicEKE, prob; nsteps)
                  E₁ = Diagnostic(Utils.FirstBaroclinicEKE, prob; nsteps)
                  EKE = Diagnostic(Utils.FullEKE, prob; nsteps)

                  # Diffusivity
                  D₁ = Diagnostic(Utils.FirstBaroclinicDiffusivity, prob; nsteps)
                  D = Diagnostic(Utils.PVDiffusivity, prob; nsteps)
                        
                  # Mixing length
                  l₁ = Diagnostic(Utils.FirstBaroclinicMixingLength, prob; nsteps)
                  l = Diagnostic(Utils.PVMixingLength, prob; nsteps)

                  diags = [
                        E₀,
                        E₁,
                        EKE,
                        D₁,
                        D,
                        l₁,
                        l
                        ]
            end

            # Step forward until next diagnostic save
            stepforward!(prob, diags, nsubs_diags)
            MultiLayerQG.updatevars!(prob)
            
            # Save at diagnostic frequency
            saveoutput(out_diags)

            # Save at 3D field frequency
            if j % Int(dtsnap_fields / dtsnap_diags) == 0

                  saveoutput(out_fields)
            end
      end
end

      ### Get real space solution ###

function get_q(prob)
      sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

      # We want to save CPU arrays not GPU arrays
      A = device_array(GPU())
      B = device_array(CPU())

      q = A(zeros(size(vars.q)))
      qh = prob.sol
      MultiLayerQG.invtransform!(q, qh, params)

      return B(q)
end

      ### Initialize and then call step forward function ###

function start!()
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, U, H, b, eta, μ, dt, stepper, aliased_fraction = 0)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

      ### Set initial condition from restart ###
      # Find most recent .nc file and use as restart

      folder = "../../output"
      ext = "_fields.nc"
      files = filter(f -> endswith(f, ext), readdir(folder, join = true))
      restart_path = isempty(files) ? nothing :
      argmax(files) do f
            stat(f).mtime  # modification time
      end

      # Set initial condition as last output in restart file
      A = device_array(grid.device)
      file = NCDataset(restart_path, "r")
      qi = file["q"][:, :, :, end]
      MultiLayerQG.set_q!(prob, A(qi))
      clock.t = file["t"][end]
      close(file)

      ### Define diagnostics ###
      # Energies
      E₀ = Diagnostic(Utils.BarotropicEKE, prob; nsteps)
      E₁ = Diagnostic(Utils.FirstBaroclinicEKE, prob; nsteps)
      EKE = Diagnostic(Utils.FullEKE, prob; nsteps)

      # Diffusivity
      D₁ = Diagnostic(Utils.FirstBaroclinicDiffusivity, prob; nsteps)
      D = Diagnostic(Utils.PVDiffusivity, prob; nsteps)
      
      # Mixing length
      l₁ = Diagnostic(Utils.FirstBaroclinicMixingLength, prob; nsteps)
      l = Diagnostic(Utils.PVMixingLength, prob; nsteps)

      diags = [
            E₀,
            E₁,
            EKE,
            D₁,
            D,
            l₁,
            l
            ]

      filename_fields = Params.path_name[1:end-5] * "_fields" * ".jld2"
      if isfile(filename_fields); rm(filename_fields); end

      filename_diags = Params.path_name[1:end-5] * "_diags" * ".jld2"
      if isfile(filename_diags); rm(filename_diags); end

      # Output 3D fields
      out_fields = Output(prob, filename_fields, (:q, get_q))

      # Output diagnostics
      out_diags = Output(prob, filename_diags,
                  (:EKE, Utils.FullEKE),
                  (:E₀, Utils.BarotropicEKE),
                  (:E₁, Utils.FirstBaroclinicEKE),
                  (:D₁, Utils.FirstBaroclinicDiffusivity),
                  (:D, Utils.PVDiffusivity),
                  (:l₁, Utils.FirstBaroclinicMixingLength),
                  (:l, Utils.PVMixingLength)
                  )

      simulate!(prob, grid, diags, EKE, out_fields, out_diags, tmax, nsteps, dtsnap_diags, dtsnap_fields, nsubs_diags, nsubs_fields)
end