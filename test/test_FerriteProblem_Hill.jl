
using Ferrite, Tensors, SparseArrays, LinearAlgebra
using FerriteProblems, FESolvers, FerriteAssembly
import FerriteProblems as FP
# import FerriteAssembly.ExampleElements: J2Plasticity
using Plots; gr();
using ProgressMeter

import MaterialModels as MM
# using FerriteNeumann

struct VectorRamp{dim,T}<:Function
    ramp::Vec{dim,T}
end

(vr::VectorRamp)(x, t, n) = t*vr.ramp

const traction_function = VectorRamp(Vec(0.0, 0.0, 1.e7))

yieldStress(ϵ) = 376.9*(0.0059+ϵ)^0.152


function setup_problem_definition()
    # Define material properties
    # material = J2Plasticity(;E=200.0e9, ν=0.3, σ0=200.e6, H=10.0e9)
    material = MM.PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51, yieldStress)
    ip =  Lagrange{RefTetrahedron, 1}()^3
    # CellValues
    cv = CellValues(QuadratureRule{RefTetrahedron}(2), ip)

    # Grid and degrees of freedom (`Ferrite.jl`)
    grid = generate_grid(Tetrahedron, (20,2,4), zero(Vec{3}), Vec((10.,1.,1.)))
    dh = DofHandler(grid); push!(dh, :u, ip); close!(dh)

    # Constraints (Dirichlet boundary conditions, `Ferrite.jl`)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), Returns(zero(Vec{3}))))
    close!(ch)

    # Neumann boundary conditions
    lh = LoadHandler(dh)
    quad_order = 3
    add!(lh, Neumann(:u, quad_order, getfaceset(grid, "right"), traction_function))

    domainspec = DomainSpec(dh, material, cv)
    return FEDefinition(domainspec; ch, lh)
end;

struct PlasticityPostProcess{T}
    tmag::Vector{T}
    umag::Vector{T}
end

PlasticityPostProcess() = PlasticityPostProcess(Float64[], Float64[]);

function FESolvers.postprocess!(post::PlasticityPostProcess, p, step, solver)
    # p::FerriteProblem
    # First, we save some values directly in the `post` struct
    push!(post.tmag, traction_function(zero(Vec{3}), FP.get_time(p), zero(Vec{3}))[3])
    push!(post.umag, maximum(abs, FP.getunknowns(p)))

    # Second, we save some results to file
    # * We must always start by adding the next step.
    FP.addstep!(p.io, p)
    # * Save the dof values (only displacments in this case)
    FP.savedofdata!(p.io, FP.getunknowns(p))
    # * Save the state in each integration point
    FP.saveipdata!(p.io, FP.get_state(p), "state")
end;

function plot_results(post::PlasticityPostProcess;
    plt=plot(), label=nothing, markershape=:auto, markersize=4
    )
    plot!(plt, post.umag, post.tmag, linewidth=0.5, title="Traction-displacement", label=label,
        markeralpha=0.75, markershape=markershape, markersize=markersize)
    ylabel!(plt, "Traction [Pa]")
    xlabel!(plt, "Maximum deflection [m]")
    return plt
end;

function example_solution()
    def = setup_problem_definition()

    # Fixed uniform time steps
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(;num_steps=25,Δt=0.04))
    problem = FerriteProblem(def, PlasticityPostProcess(), joinpath(pwd(), "A"))
    solve_problem!(problem, solver)
    plt = plot_results(problem.post, label="uniform", markershape=:x, markersize=5)

    # Same time steps as Ferrite example, overwrite results by specifying the same folder
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0), FixedTimeStepper(append!([0.], collect(0.5:0.05:1.0))))
    problem = FerriteProblem(def, PlasticityPostProcess(), joinpath(pwd(), "A"))
    solve_problem!(problem, solver)
    plot_results(problem.post, plt=plt, label="fixed", markershape=:circle)

    # Adaptive time stepping, save results to new folder
    ts = AdaptiveTimeStepper(0.05, 1.0; Δt_min=0.01, Δt_max=0.2)
    solver = QuasiStaticSolver(NewtonSolver(;tolerance=1.0, maxiter=6), ts)
    problem = FerriteProblem(def, PlasticityPostProcess(), joinpath(pwd(), "B"))
    solve_problem!(problem, solver)
    plot_results(problem.post, plt=plt, label="adaptive", markershape=:circle)

    plot!(;legend=:bottomright)
    return plt, problem, solver
end;

plt, problem, solver = example_solution();

