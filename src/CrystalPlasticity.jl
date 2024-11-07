# using LinearAlgebra
# using Tensors
# using NLsolve
# using Test

# Define Slip System
struct SlipSystem{T}
    n::Vector{T}  # Normal to the slip plane
    s::Vector{T}  # Slip direction
    tau_c0::T     # Initial critical resolved shear stress
    H::T          # Hardening modulus
end

# Define Crystal Plasticity material
struct CrystalPlasticity{T} <: AbstractMaterial
    Celas :: SymmetricTensor{4, 3, T, 36}
    slipsystems :: Vector{SlipSystem{T}}
end

# Define State
struct CrystalPlasticityState{T} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2, 3, T} # total plastic strain
    εᵉ::SymmetricTensor{2, 3, T} # total elastic strain
    σ :: SymmetricTensor{2, 3, T} # Cauchy stress
    γ :: Vector{T} # slip system shear strains
end

# Initialize material state
initial_material_state(m::CrystalPlasticity) = CrystalPlasticityState(zero(SymmetricTensor{2, 3}), 
    zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), zeros(length(m.slipsystems)))

# Cache for NLsolve
struct CrystalPlasticityCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

# Number of scalar equations
get_n_scalar_equations(m::CrystalPlasticity) = 6 + length(m.slipsystems)

# Residuals
struct ResidualsCrystalPlasticity{T}
    σ::SymmetricTensor{2, 3, T, 6}
    γ_dot::Vector{T}
end

# Required functions
Tensors.get_base(::Type{CrystalPlasticity{T}}) where T = ResidualsCrystalPlasticity

function get_cache(m::CrystalPlasticity)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2, 3}))), r_vector, x_vector, m)
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
    σ = frommandel(SymmetricTensor{2, 3}, view(v, 1:6))
    γ_dot = v[7:end]
    return ResidualsCrystalPlasticity{T}(σ, γ_dot)
end

function material_response(m::CrystalPlasticity, dε::SymmetricTensor{2, 3, T, 6}, state::CrystalPlasticityState{T},
    Δt=nothing; cache=nothing, options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

    nlsolve_cache = cache.nlsolve_cache

    σ_trial = state.σ + m.Celas ⊡ dε
    Φ = maximum([abs(resolved_shear_stress(slip_system, σ_trial)) - slip_system.tau_c0 for slip_system in m.slipsystems])

    if Φ <= 0
        return σ_trial, m.Celas, CrystalPlasticityState(state.εᵖ, state.εᵉ+dε, σ_trial, state.γ)
    else
        f(r_vector, x_vector) = vector_residual!(((x)->residuals(x, m, state, dε)), r_vector, x_vector, m)
        update_cache!(nlsolve_cache, f)
        x0 = ResidualsCrystalPlasticity(σ_trial, zeros(length(m.slipsystems)))
        tomandel!(nlsolve_cache.x_f, x0)
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)

        if result.f_converged
            x = frommandel(ResidualsCrystalPlasticity, result.zero::Vector{T})
            ∂f∂σ = [Tensors.gradient((σ)->resolved_shear_stress(slip_system, σ), x.σ) for slip_system in m.slipsystems]
            dεᵖ = sum([x.γ_dot[i] * ∂f∂σ[i] for i in 1:length(∂f∂σ)])
            dεᵉ = dε - dεᵖ
            εᵖ = state.εᵖ + dεᵖ
            Cep = m.Celas # This should be corrected based on the specific slip systems
            return x.σ, Cep, CrystalPlasticityState(εᵖ, state.εᵉ+dεᵉ, x.σ, state.γ + x.γ_dot)
        else
            error("Material model not converged. Could not find material state.")
        end

    end

end

function residuals(vars::ResidualsCrystalPlasticity, m::CrystalPlasticity, state::CrystalPlasticityState, dε)

    dεᵖ = sum([vars.γ_dot[i] * Tensors.gradient((σ)->resolved_shear_stress(slip_system, σ), vars.σ) for i in 1:length(vars.γ_dot)])
    Rσ = vars.σ - state.σ + m.Celas ⊡ (dεᵖ - dε)
    Rγ = [vars.γ_dot[i] - resolved_shear_stress(m.slipsystems[i], vars.σ) for i in 1:length(vars.γ_dot)]

    return ResidualsCrystalPlasticity(Rσ, Rγ)
end

# Helper functions
function resolved_shear_stress(slip_system::SlipSystem, σ::SymmetricTensor{2, 3})
    return dot(slip_system.s, σ * slip_system.n)
end

# Example uniaxial test
# @testset begin
    using Plots

    E = 69e3
    ν = 0.3
    Celas = elastic_tangent_3D(E, ν)

    slip_systems = [SlipSystem([1., 0., 0.], [0., 1., 0.], 5000.0, 1000.0), 
                    SlipSystem([0., 1., 0.], [1., 0., 0.], 5000.0, 1000.0)]
    m = CrystalPlasticity(Celas, slip_systems)

    function uniaxialTest(m, loadingRange, Δε; num_cycles=1)
        cache = get_cache(m)
        state = initial_material_state(m)
        e_all = [0.0]
        s_all = [0.0]
        ∂σ∂ε = zeros(SymmetricTensor{4, 3})

        for cycle in 1:num_cycles
            for e11 in loadingRange
                σ, ∂σ∂ε, state = material_response(m, Δε, state; cache=cache)
                push!(e_all, e11)
                push!(s_all, σ[1, 1])
            end
            for e11 in reverse(loadingRange)
                σ, ∂σ∂ε, state = material_response(m, -Δε, state; cache=cache)
                push!(e_all, e11)
                push!(s_all, σ[1, 1])
            end
        end

        return e_all, s_all, ∂σ∂ε, state
    end

    loadingRange = range(0.0, 0.1, 201)
    Δε = SymmetricTensor{2, 3, Float64}((i, j) -> i == 1 && j == 1 ? loadingRange.step.hi : 0.0)

    e_all, s_all, ∂σ∂ε, state = uniaxialTest(m, loadingRange, Δε, num_cycles=4)
    p = plot(e_all, s_all, xminorgrid=:on, yminorgrid=:on)
# end
