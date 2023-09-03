
yieldStress(ϵ) = 376.9*(0.0059+ϵ)^0.152

m = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51, yieldStress) # Swift hardening rule


cache = get_cache(m)

state = initial_material_state(m)

# elastic branch
initial_yield_stress = m.yieldStress(0.0)
ε11_yield = initial_yield_stress / m.E

Δε = SymmetricTensor{2,3,Float64}((i,j)-> i==1 && j==1 ? 0.5*ε11_yield : (i == j ? -0.5ε11_yield*m.ν : 0.0)')

σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)
@test σ ≈ SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j == 1 ? 0.5ε11_yield*m.E : 0.0)

# plastic branch
Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 2ε11_yield : (i == j ? -2ε11_yield*m.ν : 0.0))
σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)

Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? 5 * ε11_yield : 0.0)
σ, ∂σ∂ε, temp_state = material_response(m, Δε, state; cache=cache)


function uniaxialTest(loadingRange)
    yieldStress(ϵ) = 376.9*(0.0059+ϵ)^0.152
    m = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51, yieldStress) # Swift hardening rule
    cache = get_cache(m)
    state = initial_material_state(m)
    e11_all = Float64[]
    s11_all = Float64[]

    for e11 in loadingRange
        Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)
        σ, ∂σ∂ε, state = material_response(m, Δε, state; cache=cache)
        push!(e11_all, e11)
        push!(s11_all, σ[1,1])
    end
    return e11_all, s11_all
end

e11_all, s11_all = uniaxialTest(range(0.0, 0.05, 21))

using Plots

p = plot(e11_all, s11_all)


# stress_test = zero(SymmetricTensor{2,3})

# yield_func(stress_test)

# df_dσ = Tensors.gradient(yield_func, stress_test)

# ̇εₚ = ̇λ*∂f∂σ, where λ=̄εₚ (equivalnet plastic strain), Hardening Law: Y = K*εⁿ (yield stress!)
# λ = sqrt(2/3*dev(εₚ)⊡dev(εₚ))


function get_Plastic_loading()
    strain1 = range(0.0,  0.005, length=5)
    strain2 = range(0.005, 0.001, length=5)
    strain3 = range(0.001, 0.007, length=5)

    _C = [strain1[2:end]..., strain2[2:end]..., strain3[2:end]...]
    ε = [SymmetricTensor{2,3}((x, x/10, 0.0, 0.0, 0.0, 0.0)) for x in _C]

    return ε
end  


# @testset "PlasticHill" begin
    
#     m = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51)
#     cache = get_cache(m)

#     # initial state
#     state = initial_material_state(m)

#     # elastic branch
#     Δε = SymmetricTensor{2,3,Float64}((i,j)-> i==1 && j==1 ? 0.5)

# end


@testset "PlasticHill jld2" begin
    m = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51, yieldStress)
    
    loading = get_Plastic_loading()
    check_jld2(m, loading, "Plastic1")#, debug_print=true, OVERWRITE_JLD2=true)
end