using Ferrite, Tensors, SparseArrays, LinearAlgebra, Printf
using MaterialModels
using WriteVTK

# For later use, during the post-processing step, we define a function to
# compute the von Mises effective stress.
function vonMises(σ)
    s = dev(σ)
    return sqrt(3.0/2.0 * s ⊡ s)
end;

# ## FE-problem
# What follows are methods for assembling and and solving the FE-problem.
function create_values(interpolation)
    ## setup quadrature rules
    qr      = QuadratureRule{RefTetrahedron}(2)
    facet_qr = FacetQuadratureRule{RefTetrahedron}(3)

    ## cell and facetvalues for u
    cellvalues_u = CellValues(qr, interpolation)
    facetvalues_u = FacetValues(facet_qr, interpolation)

    return cellvalues_u, facetvalues_u
end;

# ### Add degrees of freedom
function create_dofhandler(grid, interpolation)
    dh = DofHandler(grid)
    add!(dh, :u, interpolation) # add a displacement field with 3 components
    close!(dh)
    return dh
end

# ### Boundary conditions
function create_bc(dh, grid)
    dbcs = ConstraintHandler(dh)
    ## Clamped on the left side
    dofs = [1, 2, 3]
    dbc = Dirichlet(:u, getfacetset(grid, "left"), (x,t) -> [0.0, 0.0, 0.0], dofs)
    add!(dbcs, dbc)
    close!(dbcs)
    return dbcs
end;


# ### Assembling of element contributions
#
# * Residual vector `r`
# * Tangent stiffness `K`
function doassemble!(K::SparseMatrixCSC, r::Vector, cellvalues::CellValues, dh::DofHandler,
                     material, Δu, states, states_old, cache)
    assembler = start_assemble(K, r)
    nu = getnbasefunctions(cellvalues)
    re = zeros(nu)     # element residual vector
    ke = zeros(nu, nu) # element tangent matrix

    for (i, cell) in enumerate(CellIterator(dh))
        fill!(ke, 0)
        fill!(re, 0)
        eldofs = celldofs(cell)
        Δue = Δu[eldofs]
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        assemble_cell!(ke, re, cell, cellvalues, material, Δue, state, state_old, cache)
        assemble!(assembler, eldofs, re, ke)
    end
    return K, r
end

# Compute element contribution to the residual and the tangent.
#md # !!! note
#md #     Due to symmetry, we only compute the lower half of the tangent
#md #     and then symmetrize it.
#md #
function assemble_cell!(Ke, re, cell, cellvalues, material,
                        Δue, state, state_old, cache)
    n_basefuncs = getnbasefunctions(cellvalues)
    reinit!(cellvalues, cell)

    for q_point in 1:getnquadpoints(cellvalues)
        ## For each integration point, compute stress and material stiffness
        Δϵ = function_symmetric_gradient(cellvalues, q_point, Δue) # incremental strain
        # Δϵ = ϵ - state_old[q_point].εᵖ - state_old[q_point].εᵉ
        # σ, D, state[q_point] = compute_stress_tangent(ϵ, material, state_old[q_point])
        σ, D, state[q_point] = material_response(material, Δϵ, state_old[q_point]; cache=cache)

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
          ∂Ni = shape_symmetric_gradient(cellvalues, q_point, i)
            re[i] += (∂Ni ⊡ σ) * dΩ # add internal force to residual
            for j in 1:i # loop only over lower half
                ∂Nj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += ∂Ni ⊡ D ⊡ ∂Nj * dΩ
            end
        end
    end
    symmetrize_lower!(Ke)
end

# Helper function to symmetrize the material tangent
function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end;

function doassemble_neumann!(r, dh, faceset, facetvalues, t)
    n_basefuncs = getnbasefunctions(facetvalues)
    re = zeros(n_basefuncs)                      # element residual vector
    for fc in FacetIterator(dh, faceset)
        ## Add traction as a negative contribution to the element residual `re`:
        reinit!(facetvalues, fc)
        fill!(re, 0)
        for q_point in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, q_point)
            for i in 1:n_basefuncs
                δu = shape_value(facetvalues, q_point, i)
                re[i] -= (δu ⋅ t) * dΓ
            end
        end
        assemble!(r, celldofs(fc), re)
    end
    return r
end


yieldStress(ϵ) = 376.9*(0.0059+ϵ)^0.152
    

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    M::T
    N::T
    L::T
end 

function (f::Yield_Hill48)(T::SymmetricTensor{2,3})
    return sqrt( (f.F*T[1,1]*T[1,1] + f.G*T[2,2]*T[2,2] + f.H*T[3,3]*T[3,3])/(f.F*f.G + f.F*f.H + f.G*f.H) + 2*T[2,3]*T[2,3]/f.L + 2*T[3,1]*T[3,1]/f.M + 2*T[1,2]*T[1,2]/f.N )
end


# Define a function which solves the FE-problem.
function solve()
    ## Define material parameters
    # E = 200.0e9 # [Pa]
    # H = E/20   # [Pa]
    # ν = 0.3     # [-]
    # σ₀ = 200e6  # [Pa]
    # material = J2Plasticity(E, ν, σ₀, H)

    E = 69e3
    ν = 0.3
    Celas = elastic_tangent_3D(E, ν)

    R0 =  0.84
    R45 = 0.64
    R90 = 1.51

    F = R0/(R90*(R0+1))
    G = 1/(1+R0)
    H = R0/(1+R0)
    N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
    L = N # must be checked!
    M = N # must be checked!

    yield_Hill48 = Yield_Hill48(F, G, H, M, N, L)
    yieldFunction1(T::SymmetricTensor{2,3}) = yield_Hill48(T)

    material = GeneralPlastic(Celas, yieldStress, yieldFunction1)

    L = 10.0 # beam length [m]
    w = 1.0  # beam width [m]
    h = 1.0  # beam height[m]
    n_timesteps = 20
    u_max = zeros(n_timesteps)
    traction_magnitude = 1.e1 * range(0.0, 1.0, length=n_timesteps)

    ## Create geometry, dofs and boundary conditions
    n = 2
    nels = (10n, n, 2n) # number of elements in each spatial direction
    P1 = Vec((0.0, 0.0, 0.0))  # start point for geometry
    P2 = Vec((L, w, h))        # end point for geometry
    grid = generate_grid(Tetrahedron, nels, P1, P2)
    interpolation = Lagrange{RefTetrahedron, 1}()^3

    dh = create_dofhandler(grid, interpolation) # JuaFEM helper function
    dbcs = create_bc(dh, grid) # create Dirichlet boundary-conditions

    cellvalues, facetvalues = create_values(interpolation)

    ## Pre-allocate solution vectors, etc.
    n_dofs = ndofs(dh)  # total number of dofs
    u_new  = zeros(n_dofs)  # solution vector
    u_old  = zeros(n_dofs)
    Δu = zeros(n_dofs)  # displacement correction
    r = zeros(n_dofs)   # residual
    K = allocate_matrix(dh); # tangent stiffness matrix

    ## Create material states. One array for each cell, where each element is an array of material-
    ## states - one for each integration point
    nqp = getnquadpoints(cellvalues)
    states = [initial_material_state(material) for _ in 1:nqp, _ in 1:getncells(grid)]
    states_old = [initial_material_state(material) for _ in 1:nqp, _ in 1:getncells(grid)]

    cache = get_cache(material)

    ## Newton-Raphson loop
    NEWTON_TOL = 1e-2 # 1 N
    print("\n Starting Netwon iterations:\n")

    for timestep in 1:n_timesteps
        t = timestep # actual time (used for evaluating d-bndc)
        traction = Vec((0.0, 0.0, traction_magnitude[timestep]))
        newton_itr = -1
        print("\n Time step @time = $timestep:\n")
        update!(dbcs, t) # evaluates the D-bndc at time t
        apply!(u_new, dbcs)  # set the prescribed values in the solution vector
        # Δu = zeros(n_dofs)
        Δu .= u_new - u_old
        while true; newton_itr += 1

            if newton_itr > 100
                error("Reached maximum Newton iterations, aborting")
                break
            end
            ## Tangent and residual contribution from the cells (volume integral)
            doassemble!(K, r, cellvalues, dh, material, Δu, states, states_old, cache);
            ## Residual contribution from the Neumann boundary (surface integral)
            doassemble_neumann!(r, dh, getfacetset(grid, "right"), facetvalues, traction)
            norm_r = norm(r[Ferrite.free_dofs(dbcs)])

            print("Iteration: $newton_itr \tresidual: $(@sprintf("%.8f", norm_r))\n")
            if norm_r < NEWTON_TOL
                break
            end

            apply_zero!(K, r, dbcs)
            dΔu = -Symmetric(K) \ r
        
            Δu += dΔu
        end

        u_new .= u_old + Δu

        u_old .= u_new
        
        ## Update the old states with the converged values for next timestep
        states_old .= states

        u_max[timestep] = maximum(abs, u_new) # maximum displacement in current timestep

        ## ## Postprocessing
        ## Only a vtu-file corresponding to the last time-step is exported.
        ##
        ## The following is a quick (and dirty) way of extracting average cell data for export.
        mises_values = zeros(getncells(grid))
        λ_values = zeros(getncells(grid))
        ϵp = zeros(SymmetricTensor{2,3}, getncells(grid))
        for (el, cell_states) in enumerate(eachcol(states))
            for state in cell_states
                mises_values[el] += vonMises(state.σ)
                λ_values[el] += state.λ
                ϵp[el] += state.εᵖ
            end
            mises_values[el] /= length(cell_states) # average von Mises stress
            λ_values[el] /= length(cell_states)     # average drag stress
            ϵp[el] /= length(cell_states)
        end
        EquiEp = [vonMises(ϵp[el]) for el in 1:getncells(grid)]
        VTKGridFile("test/Results/plasticity-"*string(timestep), dh) do vtk
            write_solution(vtk, dh, u_new) # displacement field
            write_cell_data(vtk, mises_values, "von Mises [MPa]")
            write_cell_data(vtk, λ_values, "Equi_plastic")
            write_cell_data(vtk, EquiEp, "Vm_Ep")
        end

    end

    return u_max, traction_magnitude
end

# Solve the FE-problem and for each time-step extract maximum displacement and
# the corresponding traction load. Also compute the limit-traction-load
u_max, traction_magnitude = solve();

# Finally we plot the load-displacement curve.
using Plots
plot(
    vcat(0.0, u_max),                # add the origin as a point
    vcat(0.0, traction_magnitude),
    linewidth=2,
    title="Traction-displacement",
    label=nothing,
    markershape=:circle
    )
ylabel!("Traction [MPa]")
xlabel!("Maximum deflection [m]")
