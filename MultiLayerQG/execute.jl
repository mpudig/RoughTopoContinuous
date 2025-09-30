# This is the driver: set up, run, and save the model

# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, Printf, JLD2, NCDatasets, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays, KernelAbstractions;
using FourierFlows: CPU, GPU

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

f₀ = Params.f0
β = Params.beta
g = Params.g
μ = Params.μ

b = Params.b
U = Params.U

#eta = Params.eta

      ### Time stepping ###

dt = Params.dt
nsubs = Params.nsubs
nsteps = Params.nsteps
dtsnap = Params.dtsnap
tmax = Params.tmax
stepper = Params.stepper

      ### Step the model forward ###

function simulate!(nsteps, nsubs, dtsnap, tmax, grid, prob, out, diags, EKE)
      saveproblem(out)
      saveoutput(out)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs)

      for j = frames
      cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            if j % 5 == 0

                  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, KE1: %.3e, KEN: %.3e, walltime: %.2f min",
                  clock.step, clock.t, cfl, EKE.data[EKE.i][1][1], EKE.data[EKE.i][1][end], (time()-startwalltime)/60)

                  println(log)
                  flush(stdout)
            end

            # If cfl is close to unstable value, halve the time step and reset the problem with initial condition at last time step
            if cfl > 0.85

                  # Reset time stepping variables
                  dt = clock.dt / 2
                  clock.dt = dt
                  nsubs = Int(dtsnap / dt)
                  nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs) * nsubs

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

                  diags = [E₀,
                        E₁,
                        EKE,
                        D₁,
                        D,
                        l₁,
                        l
                        ]
            end

            stepforward!(prob, diags, nsubs)
            MultiLayerQG.updatevars!(prob)
            saveoutput(out)
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
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, U, H, b, μ, dt, stepper, aliased_fraction = 0)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      x, y = grid.x, grid.y

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

      diags = [E₀,
               E₁,
               EKE,
               D₁,
               D,
               l₁,
               l
               ]

      filename = Params.path_name
      if isfile(filename); rm(filename); end

      # Output diagnostics
      out = Output(prob, filename,
                  (:q, get_q),
                  (:EKE, Utils.FullEKE),
                  (:E₀, Utils.BarotropicEKE),
                  (:E₁, Utils.FirstBaroclinicEKE),
                  (:D₁, Utils.FirstBaroclinicDiffusivity),
                  (:D, Utils.PVDiffusivity),
                  (:l₁, Utils.FirstBaroclinicMixingLength),
                  (:l, Utils.PVMixingLength)
                  )

      Utils.set_initial_condition!(prob, Params.E0, Params.K0, Params.Kd, Params.ϕ₁)

      simulate!(nsteps, nsubs, dtsnap, tmax, grid, prob, out, diags, EKE)
end