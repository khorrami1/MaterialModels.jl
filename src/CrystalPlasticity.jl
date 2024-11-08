using LinearAlgebra
using Tensors
using NLsolve
using Plots

function elastic_tensor_aluminum()
    C11 = 106.75  # GPa
    C12 = 60.41   # GPa
    C44 = 28.34   # GPa

    # Initialize the elastic tensor with specific values for aluminum
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


# Define FCC Slip Systems with explicit type Float64
function fcc_slip_systems()
    # 12 slip systems for FCC
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

    slip_systems = SlipSystem{Float64}[]  # Explicitly specify Float64
    for i in 1:12
        n = normalize(Vector{Float64}(slip_normals[i]))
        s = normalize(Vector{Float64}(slip_directions[i]))
        push!(slip_systems, SlipSystem(n, s, 20.0, 100.0))  # Initial CRSS (20 MPa), Hardening modulus (100 MPa)
    end
    return slip_systems
end

# Define Slip System Structure
struct SlipSystem{T}
    n::Vector{T}    # Slip plane normal
    s::Vector{T}    # Slip direction
    tau_c0::T       # Initial critical resolved shear stress (CRSS)
    H::T            # Hardening modulus
end

# Define Crystal Plasticity Material
struct CrystalPlasticity{T} <: AbstractMaterial
    Celas :: SymmetricTensor{4, 3, T, 36}
    slipsystems :: Vector{SlipSystem{T}}
end

# Define State
struct CrystalPlasticityState{T} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2, 3, T}  # Plastic strain tensor
    εᵉ::SymmetricTensor{2, 3, T}  # Elastic strain tensor
    σ :: SymmetricTensor{2, 3, T} # Cauchy stress tensor
    γ :: Vector{T}                # Slip system shear strains
end

# Initialize Material State
function initial_material_state(m::CrystalPlasticity)
    return CrystalPlasticityState(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), zeros(length(m.slipsystems)))
end

# Resolved Shear Stress Calculation
function resolved_shear_stress(slip_system::SlipSystem, σ::SymmetricTensor{2, 3})
    return dot(slip_system.s, σ * slip_system.n)
end

# Material Response for Crystal Plasticity
function material_response(m::CrystalPlasticity, dε::SymmetricTensor{2, 3, T, 6}, state::CrystalPlasticityState{T}; Δt=nothing) where T
    σ_trial = state.σ + m.Celas ⊡ dε
    Φ = maximum([abs(resolved_shear_stress(slip_system, σ_trial)) - slip_system.tau_c0 for slip_system in m.slipsystems])

    if Φ <= 0
        return σ_trial, m.Celas, CrystalPlasticityState(state.εᵖ, state.εᵉ+dε, σ_trial, state.γ)
    else
        # Plastic update step with Newton-Raphson iterations
        γ_dot = zeros(length(m.slipsystems))
        for i in 1:10  # Maximum 10 iterations
            Rσ = σ_trial - state.σ + m.Celas ⊡ (dε - sum([γ_dot[i] * Tensors.gradient((σ)->resolved_shear_stress(m.slipsystems[i], σ), σ_trial) for i in 1:length(m.slipsystems)]))
            γ_dot = update_slip_rates(m, state, σ_trial, γ_dot)  # Update with hardening
            if norm(Rσ) < 1e-6
                break
            end
        end
        # Update state
        return σ_trial, m.Celas, CrystalPlasticityState(state.εᵖ, state.εᵉ + dε, σ_trial, state.γ + γ_dot)
    end
end

# Update Slip Rates Considering Hardening
function update_slip_rates(m::CrystalPlasticity, state::CrystalPlasticityState, σ_trial, γ_dot)
    for (i, slip_system) in enumerate(m.slipsystems)
        τ_res = resolved_shear_stress(slip_system, σ_trial)
        γ_dot[i] = max(0, (abs(τ_res) - (slip_system.tau_c0 + slip_system.H * state.γ[i])) / slip_system.H)
    end
    return γ_dot
end

# Run Uniaxial Test
function uniaxialTest(m, loadingRange, Δε)
    state = initial_material_state(m)
    e_all, s_all = [], []
    for e11 in loadingRange
        σ, _, state = material_response(m, Δε, state)
        push!(e_all, e11)
        push!(s_all, σ[1, 1])
    end
    return e_all, s_all
end

# Main Setup
# Create the CrystalPlasticity instance with explicit Float64
Celas = elastic_tensor_aluminum()
slip_systems = fcc_slip_systems()
m = CrystalPlasticity{Float64}(Celas, slip_systems)  # Specify the type parameter here

# Define loading for uniaxial test
loadingRange = range(0.0, 0.01, length=100)
Δε = SymmetricTensor{2, 3, Float64}((i, j) -> i == 1 && j == 1 ? Float64(loadingRange.step) : 0.0)

# Run test
e_all, s_all = uniaxialTest(m, loadingRange, Δε)
plot(e_all, s_all, xlabel="Strain", ylabel="Stress (GPa)", title="Uniaxial Test for FCC Aluminum Crystal Plasticity", lw=2)
