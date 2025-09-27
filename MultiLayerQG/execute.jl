# This is the driver: set up, run, and save the model

# include all modules
include("utils.jl")
include("params.jl")

# compile other packages
using GeophysicalFlows, FFTW, Statistics, Random, Printf, JLD2, NCDatasets, CUDA, CUDA_Driver_jll, CUDA_Runtime_jll, GPUCompiler, GPUArrays, NCDatasets;

# local import
import .Utils
import .Params



      ### Save path and device ###

path_name = Params.path_name
dev = Params.dev



      ### Grid ###

nx = Params.nx
Lx = Params.Lx

nlayers = Params.nz
H = Params.H



      ### Planetary parameters ###

f₀ = Params.f0
β = Params.beta
g = Params.g
μ = Params.kappa

ρ = Params.rho
U = Params.U

eta = Params.eta



      ### Time stepping ###

dt = Params.dt
nsubs = Params.nsubs
nsteps = Params.nsteps
stepper = Params.stepper



      ### Step the model forward ###

function simulate!(nsteps, nsubs, grid, prob, out, diags, KE, APE)
      saveproblem(out)
      saveoutput(out)
      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      
      startwalltime = time()
      frames = 0:round(Int, nsteps / nsubs)
      
      for j = frames
            if j % 5 == 0
                  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
      
                  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min",
                   clock.step, clock.t, cfl, KE.data[KE.i][1], KE.data[KE.i][2], APE.data[APE.i][1], (time()-startwalltime)/60)
      
                  println(log)
                  flush(stdout)
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
      prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, β, g, U, H, ρ, eta, μ, 
                                  dt, stepper, aliased_fraction=0)

      sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
      x, y = grid.x, grid.y
      A = device_array(grid.device)

      KE = Diagnostic(Utils.calc_KE, prob; nsteps)
      APE = Diagnostic(Utils.calc_APE, prob; nsteps)
      D = Diagnostic(Utils.calc_meridiff, prob; nsteps)
      V = Diagnostic(Utils.calc_meribarovel, prob; nsteps)
      Lmix = Diagnostic(Utils.calc_mixlen, prob; nsteps)

      #KEFlux1 = Diagnostic(Utils.calc_KEFlux_1, prob; nsteps)
      #APEFlux1 = Diagnostic(Utils.calc_APEFlux_1, prob; nsteps)
      #ShearFlux1 = Diagnostic(Utils.calc_ShearFlux_1, prob; nsteps)
      #KEFlux2 = Diagnostic(Utils.calc_KEFlux_2, prob; nsteps)
      #APEFlux2 = Diagnostic(Utils.calc_APEFlux_2, prob; nsteps)
      #TopoFlux2 = Diagnostic(Utils.calc_TopoFlux_2, prob; nsteps)
      #DragFlux2 = Diagnostic(Utils.calc_DragFlux_2, prob; nsteps)

      diags = [KE, APE, D, V, Lmix]#, KEFlux1, APEFlux1, ShearFlux1, KEFlux2, APEFlux2, TopoFlux2, DragFlux2]

      filename = Params.path_name
      if isfile(filename); rm(filename); end

      out = Output(prob, filename, (:q, get_q),
                  (:KE, Utils.calc_KE), (:APE, Utils.calc_APE),
                  (:D, Utils.calc_meridiff), (:V, Utils.calc_meribarovel), (:Lmix, Utils.calc_mixlen))#,
                  #(:KEFlux1, Utils.calc_KEFlux_1),
                  #(:APEFlux1, Utils.calc_APEFlux_1),
                  #(:ShearFlux1, Utils.calc_ShearFlux_1),
                  #(:KEFlux2, Utils.calc_KEFlux_2),
                  #(:APEFlux2, Utils.calc_APEFlux_2),
                  #(:TopoFlux2, Utils.calc_TopoFlux_2),
                  #(:DragFlux2, Utils.calc_DragFlux_2)
                  #)

      # If starting from t = 0:
      Utils.set_initial_condition!(prob, Params.E0, Params.K0, Params.Kd)

      # If starting as a restart:
      #ds = NCDataset("../../output" * Params.expt_name * ".nc", "r")
      #qi = ds["q"][:, :, :, end]
      #MultiLayerQG.set_q!(prob, A(qi))
      #close(ds)

      simulate!(nsteps, nsubs, grid, prob, out, diags, KE, APE)
end