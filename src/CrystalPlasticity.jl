using LinearAlgebra
using Tensors
using NLsolve
using Plots

# Define Yield Function and Yield Stress (as placeholders)
function YieldFunction(slip_system::SlipSystem, σ::SymmetricTensor{2,3})
    return abs(dot(slip_system.s, σ * slip_system.n))  # Resolved shear stress
end

# Define Voce Hardening Law for CRSS
function YieldStress(slip_system::SlipSystem, γ::Float64)
    τ_sat = 150.0  # Saturation stress [MPa] (example value for Aluminum)
    h = 50.0       # Hardening rate [MPa] (example value)
    return slip_system.tau_c0 + (τ_sat - slip_system.tau_c0) * (1 - exp(-h * γ))
end

# Define Slip System
struct SlipSystem{T}
    n::Vector{T}  # Slip plane normal
    s::Vector{T}  # Slip direction
    tau_c0::T     # Initial CRSS
end

# Crystal Plasticity Material Structure
struct CrystalPlasticity{T} <: AbstractMaterial
    Celas :: SymmetricTensor{4, 3, T, 36}
    slipsystems :: Vector{SlipSystem{T}}
end

# Crystal Plasticity State Structure
struct CrystalPlasticityState{T} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,3,T}  # Plastic strain tensor
    εᵉ::SymmetricTensor{2,3,T}  # Elastic strain tensor
    σ :: SymmetricTensor{2,3,T} # Cauchy stress tensor
    γ :: Vector{T}              # Shear strain on each slip system
end

# Initialize Material State
function initial_material_state(m::CrystalPlasticity)
    return CrystalPlasticityState(zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), zeros(length(m.slipsystems)))
end

# Cache for NLsolve
struct CrystalPlasticityCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

get_n_scalar_equations(m::CrystalPlasticity) = 6 + length(m.slipsystems)

# Residuals for Crystal Plasticity
struct ResidualsCrystalPlasticity{T}
    σ::SymmetricTensor{2,3,T,6}
    γ_dot::Vector{T}
end

Tensors.get_base(::Type{CrystalPlasticity{T}}) where T = ResidualsCrystalPlasticity # needed for frommandel

# Initialize Cache
function get_cache(m::CrystalPlasticity)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x) -> residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return CrystalPlasticityCache(cache)
end

function Tensors.tomandel!(v::Vector{T}, r::ResidualsCrystalPlasticity{T}) where T
    tomandel!(view(v, 1:6), r.σ)
    v[7:end] = r.γ_dot
    return v
end

function Tensors.frommandel(::Type{ResidualsCrystalPlasticity}, v::Vector{T}) where T
    σ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    γ_dot = v[7:end]
    return ResidualsCrystalPlasticity{T}(σ, γ_dot)
end

# Material Response
function material_response(m::CrystalPlasticity, dε::SymmetricTensor{2,3,T,6}, state::CrystalPlasticityState{T};
    Δt=nothing, cache=nothing, options=Dict{Symbol, Any}()) where T

    nlsolve_cache = cache.nlsolve_cache

    # Trial stress
    σ_trial = state.σ + m.Celas ⊡ dε
    Φ = maximum([YieldFunction(slip_system, σ_trial) - YieldStress(slip_system, state.γ[i]) for (i, slip_system) in enumerate(m.slipsystems)])

    if Φ <= 0
        # Elastic step
        return σ_trial, m.Celas, CrystalPlasticityState(state.εᵖ, state.εᵉ + dε, σ_trial, state.γ)
    else
        # Plastic step using Newton-Raphson
        f(r_vector, x_vector) = MaterialModels.vector_residual!(((x) -> residuals(x, m, state, dε)), r_vector, x_vector, m)
        MaterialModels.update_cache!(nlsolve_cache, f)
        x0 = ResidualsCrystalPlasticity(σ_trial, zeros(length(m.slipsystems)))
        tomandel!(nlsolve_cache.x_f, x0)
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)

        if result.f_converged
            x = frommandel(ResidualsCrystalPlasticity, result.zero)
            ∂f∂σ = [Tensors.gradient((σ) -> YieldFunction(slip_system, σ), x.σ) for slip_system in m.slipsystems]
            dεᵖ = sum([x.γ_dot[i] * ∂f∂σ[i] for i in 1:length(∂f∂σ)])
            dεᵉ = dε - dεᵖ
            εᵖ = state.εᵖ + dεᵖ
            Cep = m.Celas  # Effective elastic-plastic stiffness
            return x.σ, Cep, CrystalPlasticityState(εᵖ, state.εᵉ + dεᵉ, x.σ, state.γ + x.γ_dot)
        else
            error("Material model not converged. Could not find material state.")
        end
    end
end

# Residuals Function
function residuals(vars::ResidualsCrystalPlasticity, m::CrystalPlasticity, state::CrystalPlasticityState, dε)
    dεᵖ = sum([vars.γ_dot[i] * Tensors.gradient((σ) -> YieldFunction(m.slipsystems[i], σ), vars.σ) for i in 1:length(vars.γ_dot)])
    Rσ = vars.σ - state.σ + m.Celas ⊡ (dεᵖ - dε)
    Rγ = [vars.γ_dot[i] - YieldFunction(m.slipsystems[i], vars.σ) for i in 1:length(vars.γ_dot)]
    return ResidualsCrystalPlasticity(Rσ, Rγ)
end



# Define Elastic Constants for FCC Aluminum
function elastic_tensor_aluminum()
    C11 = 106.75e3  # MPa
    C12 = 60.41e3   # MPa
    C44 = 28.34e3   # MPa

    C = SymmetricTensor{4, 3, Float64}((i, j, k, l) -> begin
        if i == j && j == k && k == l
            C11
        elseif (i == j && k == l) || (i == k && j == l) || (i == l && j == k)
            C12
        elseif (i == j && k != l) || (i == k && j != l) || (i == l && j != k)
            C44 / 2
        else
            0.0
        end
    end)
    return C
end

# Define FCC Slip Systems with Initial CRSS and Hardening Modulus
function fcc_slip_systems()
    slip_normals = [
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        [1, -1, 1], [1, -1, 1], [1, -1, 1], [1, -1, 1],
        [1, 1, -1], [1, 1, -1], [1, 1, -1], [1, 1, -1]
    ]
    slip_directions = [
        [0, 1, -1], [1, 0, -1], [0, -1, 1], [-1, 0, 1],
        [0, 1, 1], [1, 0, 1], [0, -1, -1], [-1, 0, -1],
        [0, -1, -1], [1, 0, -1], [0, 1, 1], [-1, 0, 1]
    ]

    slip_systems = SlipSystem{Float64}[]
    for i in 1:12
        n = normalize(Vector{Float64}(slip_normals[i]))
        s = normalize(Vector{Float64}(slip_directions[i]))
        push!(slip_systems, SlipSystem(n, s, 20.0))  # Initial CRSS (20 MPa), Hardening modulus (100 MPa)
    end
    return slip_systems
end

# Crystal Plasticity Material
Celas = elastic_tensor_aluminum()
slip_systems = fcc_slip_systems()
m = CrystalPlasticity{Float64}(Celas, slip_systems)

# Define a function for uniaxial loading
function uniaxial_test(m, loading_range, Δε, num_cycles=1)
    # Define the initial material state and cache
    state = initial_material_state(m)
    cache = get_cache(m)
    e_all, s_all = [], []

    for cycle in 1:num_cycles
        for e11 in loading_range
            σ, _, state = material_response(m, Δε, state; cache=cache)
            push!(e_all, e11)
            push!(s_all, σ[1, 1])
        end
    end
    return e_all, s_all
end

# Set up uniaxial loading parameters
loading_range = range(0.0, 0.01, length=100)  # Strain range for loading
Δε = SymmetricTensor{2, 3, Float64}((i, j) -> i == 1 && j == 1 ? Float64(loading_range.step) : 0.0)  # Apply strain only in the 11 direction

# Run uniaxial loading test for FCC aluminum
e_all, s_all = uniaxial_test(m, loading_range, Δε)

# Plot the stress-strain curve
plot(e_all, s_all, xlabel="Strain", ylabel="Stress [MPa]", title="Uniaxial Stress-Strain for Single Crystal Aluminum", lw=2)
