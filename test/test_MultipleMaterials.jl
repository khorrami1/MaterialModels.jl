using Ferrite, FerriteAssembly, MaterialModelsBase
# import FerriteAssembly.ExampleElements: J2Plasticity, ElasticPlaneStrain

methods(FerriteAssembly.element_routine!)

function create_grid_with_inclusion()
    p1 = Vec((-1.0, -1.0))
    p2 = Vec(( 1.0,  1.0))
    grid = generate_grid(Quadrilateral, (20,20), p1, p2)
    addcellset!(grid, "inclusion", x -> norm(x) < 0.8)
    addcellset!(grid, "matrix", setdiff(1:getncells(grid), getcellset(grid, "inclusion")))
    return grid
end
grid = create_grid_with_inclusion();

ip = Lagrange{RefQuadrilateral,1}()^2;

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), Returns(zero(Vec{2}))))
f_dbc(x, t) = Vec((0.05*t, 0.0))
add!(ch, Dirichlet(:u, getfaceset(grid, "right"), f_dbc))
close!(ch)

# Define values
qr = QuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip)

# FerriteAssembly setup
struct LinearElastic{Dim,T,N}
    C::SymmetricTensor{4,Dim,T,N}
end

# Only for plane strain
function LinearElastic(::Val{Dim}=Val(3), E=2.e3, ν=0.3) where Dim
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    I2 = one(SymmetricTensor{2,Dim})
    I4vol = I2⊗I2
    I4dev = one(SymmetricTensor{4,Dim}) - I4vol / 3
    return LinearElastic(2G*I4dev + K*I4vol)
end


"""
    J2Plasticity(;E, ν, σ0, H) <: MaterialModelsBase.AbstractMaterial

This plasticity model is taken from `Ferrite.jl`'s plasticity 
[example](https://ferrite-fem.github.io/Ferrite.jl/v0.3.14/examples/plasticity/),
and considers linear isotropic hardening, with Young's modulus, `E`, Poisson's
ratio, `ν`, initial yield limit, `σ0`, and hardening modulus, `H`. It is defined 
as an `AbstractMaterial` following the `MaterialModelsBase` interface. 
"""
struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}} <: AbstractMaterial
    G::T  # Shear modulus
    K::T  # Bulk modulus
    σ0::T # Initial yield limit
    H::T  # Hardening modulus
    D::S  # Elastic stiffness tensor
end
function J2Plasticity(;E, ν, σ0, H)
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)

    Isymdev(i,j,k,l) = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)
    temp(i,j,k,l) = 2.0G *( 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) + ν/(1.0-2.0ν)*δ(i,j)*δ(k,l))
    D = SymmetricTensor{4, 3}(temp)
    return J2Plasticity(G, K, σ0, H, D)
end

# State variables
struct J2PlasticityState{T, S <: SecondOrderTensor{3, T}} <: AbstractMaterialState
    ϵp::S # plastic strain
    k::T # hardening variable
end
function MaterialModelsBase.initial_material_state(::J2Plasticity)
    return J2PlasticityState(zero(SymmetricTensor{2,3}), 0.0)
end

# The main `material_response` function 
function MaterialModelsBase.material_response(
    material::J2Plasticity, ϵ::SymmetricTensor{2,3}, state::J2PlasticityState, Δt, cache, args...)
    ## unpack some material parameters
    G = material.G
    H = material.H

    ## We use (•)ᵗ to denote *trial*-values
    σᵗ = material.D ⊡ (ϵ - state.ϵp) # trial-stress
    sᵗ = dev(σᵗ)         # deviatoric part of trial-stress
    J₂ = 0.5 * sᵗ ⊡ sᵗ  # second invariant of sᵗ
    σᵗₑ = sqrt(3.0*J₂)   # effective trial-stress (von Mises stress)
    σʸ = material.σ0 + H * state.k # Previous yield limit

    φᵗ  = σᵗₑ - σʸ # Trial-value of the yield surface

    if φᵗ < 0.0 # elastic loading
        return σᵗ, material.D, state
    else # plastic loading
        h = H + 3G
        μ =  φᵗ / h   # plastic multiplier

        c1 = 1 - 3G * μ / σᵗₑ
        s = c1 * sᵗ           # updated deviatoric stress
        σ = s + vol(σᵗ)       # updated stress

        ## Compute algorithmic tangent stiffness ``D = \frac{\Delta \sigma }{\Delta \epsilon}``
        κ = H * (state.k + μ) # drag stress
        σₑ = material.σ0 + κ  # updated yield surface

        δ(i,j) = i == j ? 1.0 : 0.0
        Isymdev(i,j,k,l)  = 0.5*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)) - 1.0/3.0*δ(i,j)*δ(k,l)
        Q(i,j,k,l) = Isymdev(i,j,k,l) - 3.0 / (2.0*σₑ^2) * s[i,j]*s[k,l]
        b = (3G*μ/σₑ) / (1.0 + 3G*μ/σₑ)

        Dtemp(i,j,k,l) = -2G*b * Q(i,j,k,l) - 9G^2 / (h*σₑ^2) * s[i,j]*s[k,l]
        D = material.D + SymmetricTensor{4, 3}(Dtemp)

        ## Return new state
        Δϵᵖ = 3/2 * μ / σₑ * s # plastic strain
        ϵp = state.ϵp + Δϵᵖ    # plastic strain
        k = state.k + μ        # hardening variable
        return σ, D, J2PlasticityState(ϵp, k)
    end
end

function FerriteAssembly.element_routine!(Ke, re, state, ae, material::LinearElastic, cv::CellValues, buffer)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        ϵ = function_symmetric_gradient(cv, q_point, ae)
        σ = material.C ⊡ ϵ
        ## Assemble residual contributions
        for i in 1:getnbasefunctions(cv)
            ∇δNu = shape_symmetric_gradient(cv, q_point, i)
            re[i] += (∇δNu ⊡ σ )*dΩ
            ∇δNu_C = ∇δNu ⊡  material.C
            for j in 1:getnbasefunctions(cv)
                ∇Nu = shape_symmetric_gradient(cv, q_point, j)
                Ke[j,i] += (∇δNu_C ⊡ ∇Nu)*dΩ # Since Ke is symmetric, we calculate Ke' to index faster
            end
        end
    end
end

function FerriteAssembly.element_residual!(re, state, ae, material::LinearElastic, cv::CellValues, buffer)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        ϵ = function_symmetric_gradient(cv, q_point, ae)
        σ = material.C ⊡ ϵ
        ## Assemble residual contributions
        for i in 1:getnbasefunctions(cv)
            ∇δNu = shape_symmetric_gradient(cv, q_point, i)
            re[i] += (∇δNu ⊡ σ )*dΩ
        end
    end
end

elastic_material = LinearElastic(Val(2), 210e3, 0.3)
plastic_material = ReducedStressState(
    PlaneStrain(),
    J2Plasticity(;E=210e3, ν=0.3, σ0=100.0, H=10e3))

domains = Dict(
    "elastic"=>DomainSpec(dh, elastic_material, cv; set=getcellset(grid, "inclusion")),
    "plastic"=>DomainSpec(dh, plastic_material, cv; set=getcellset(grid, "matrix"))
)

buffer = setup_domainbuffers(domains);

function solve_nonlinear_timehistory(buffer, dh, ch; time_history)
    maxiter = 10
    tolerance = 1e-6
    K = create_sparsity_pattern(dh)
    r = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    # Prepare postprocessing
    pvd = paraview_collection("multiple_materials.pvd");
    stepnr = 0
    vtk_grid("multiple_materials-$stepnr", dh) do vtk
        vtk_point_data(vtk, dh, a)
        vtk_cellset(vtk, dh.grid, "inclusion")
        vtk_save(vtk)
        pvd[0.0] = vtk
    end
    for t in time_history
        # Update and apply the Dirichlet boundary conditions
        update!(ch, t)
        apply!(a, ch)
        for i in 1:maxiter
            # Assemble the system
            assembler = start_assemble(K, r)
            work!(assembler, buffer; a=a)
            # Apply boundary conditions
            apply_zero!(K, r, ch)
            # Check convergence
            norm(r) < tolerance && break
            i == maxiter && error("Did not converge")
            # Solve the linear system and update the dof vector
            a .-= K\r
            apply!(a, ch)
        end
        # Postprocess
        stepnr += 1
        vtk_grid("multiple_materials-$stepnr", dh) do vtk
            vtk_point_data(vtk, dh, a)
            vtk_cellset(vtk, dh.grid, "inclusion")
            vtk_save(vtk)
            pvd[t] = vtk
        end
        # If converged, update the old state variables to the current.
        update_states!(buffer)
    end
    vtk_save(pvd);
    return nothing
end;

solve_nonlinear_timehistory(buffer, dh, ch; time_history=collect(range(0,1,20)));