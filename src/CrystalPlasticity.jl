
slipsystems = MaterialModels.slipsystems(MaterialModels.FCC(), rand(RodriguesParam))


struct CrystalPlasticity{T} <: AbstractMaterial
end

struct CrystalPlasticityState{T} <: AbstractMaterialState
end

initial_material_state(::CrystalPlasticity)

struct CrystalPlasticityCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

get_n_scalar_equations(::CrystalPlasticity)

struct ResidualCrystalPlasticity{T}
end

Tensors.get_base(::Type{CrystalPlasticity{T}}) where T = ResidualCrystalPlasticity

function get_cache(m::CrystalPlasticity)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return CrystalPlasticityCache(cache)
end

function Tensors.tomandel!(v::Vector{T}, r::ResidualCrystalPlasticity{T}) where T
    # M=6
    # # TODO check vector length
    # tomandel!(view(v, 1:M), r.σ)
    # v[M+1] = r.κ
    # v[M+2] = r.dλ
    return v
end

function Tensors.frommandel(::Type{ResidualCrystalPlasticity}, v::Vector{T}) where T
    # σ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    # κ = v[7]
    # dλ = v[8]
    return ResidualCrystalPlasticity{T}(σ, κ, dλ)
end

# function material_response(m::GeneralPlastic, dε::SymmetricTensor{2,3,T,6}, state::GeneralPlasticState{T},
#     Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

#     nlsolve_cache = cache.nlsolve_cache

#     σ_trial = state.σ + m.Celas ⊡ dε

#     # εᵖ_equi = get_equivalent_Hill(state.εᵖ, m)
#     Φ = m.yieldFunction(σ_trial) - m.yieldStress(state.κ)

#     if Φ <= 0
#         return σ_trial, m.Celas, GeneralPlasticState(state.εᵖ, state.εᵉ+dε, σ_trial, state.κ)
#     else
#         # set the current residual function that depends only on the variables
#         # nlsolve_cache.f = (r_vector, x_vector) -> vector_residual!(((x)->residuals(x,m,state,Δε)), r_vector, x_vector, m)
#         f(r_vector, x_vector) = vector_residual!(((x)->residuals(x,m,state,dε)), r_vector, x_vector, m)
#         update_cache!(nlsolve_cache, f)
#         # initial guess
#         x0 = ResidualsGeneralPlastic(σ_trial, state.κ, 0.0)
#         # convert initial guess to vector
#         tomandel!(nlsolve_cache.x_f, x0)
#         # solve for variables x
#         nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
#         haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
#         result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)
#         println("norm(ε) = ", string(norm(state.εᵉ+ state.εᵖ))," iterations: ", string(result.iterations))
        
#         if result.f_converged
#             x = frommandel(ResidualsGeneralPlastic, result.zero::Vector{T})
#             dεᵖ = x.dλ*Tensors.gradient(m.yieldFunction, x.σ)
#             dεᵉ = dε - dεᵖ
#             εᵖ = state.εᵖ + dεᵖ
#             Cep = m.Celas # it must be corrected later!
#             return x.σ, Cep, GeneralPlasticState(εᵖ, state.εᵉ+dεᵉ, x.σ, x.κ)
#         else
#             error("Material model not converged. Could not find material state.")
#         end

#     end

# end

# function residuals(vars::ResidualsGeneralPlastic, m::GeneralPlastic, state::GeneralPlasticState, dε)

#     df_dσ = Tensors.gradient(m.yieldFunction, vars.σ)
#     dεᵖ = vars.dλ * df_dσ 
#     # εᵖ = state.εᵖ + dεᵖ
#     Rσ = vars.σ - state.σ + m.Celas ⊡ (dεᵖ - dε)
#     Rκ = vars.κ - state.κ - vars.dλ
#     # εᵖ_equi = get_equivalent_Hill(εᵖ, m)
#     RΦ = m.yieldFunction(vars.σ) - m.yieldStress(vars.κ)

#     return ResidualsGeneralPlastic(Rσ, Rκ, RΦ)
# end