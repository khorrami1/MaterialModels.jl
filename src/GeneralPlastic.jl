

# function YieldFunction() end
# function YieldStress() end

struct GeneralPlastic{T} <: AbstractMaterial
    Celas :: SymmetricTensor{4, 3, T, 36}
    # params :: Vector{T}
    yieldStress :: Function
    yieldFunction :: Function
end


struct GeneralPlasticState{T} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,3,T} # total plastic strain
    εᵉ::SymmetricTensor{2,3,T} # total elastic strain
    σ :: SymmetricTensor{2,3,T} # Cauchy stress
    κ :: T # if associative follow rule, then Equivalent plastic strain = plastic multiplier 
end

# It should be generalized for any dim1, dim2, T, and M
initial_material_state(::GeneralPlastic) = GeneralPlasticState(zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), 0.0)

struct GeneralPlasticCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

get_n_scalar_equations(::GeneralPlastic) = 8

struct ResidualsGeneralPlastic{T}
    σ::SymmetricTensor{2,3,T,6}
    κ::T
    dλ::T 
end

Tensors.get_base(::Type{GeneralPlastic{T}}) where T = ResidualsGeneralPlastic # needed for frommandel


function get_cache(m::GeneralPlastic)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return GeneralPlasticCache(cache)
end

function Tensors.tomandel!(v::Vector{T}, r::ResidualsGeneralPlastic{T}) where T
    # TODO check vector length
    tomandel!(view(v, 1:6), r.σ)
    v[7] = r.κ
    v[8] = r.dλ
    return v
end

function Tensors.frommandel(::Type{ResidualsGeneralPlastic}, v::Vector{T}) where T
    σ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    κ = v[7]
    dλ = v[8]
    return ResidualsGeneralPlastic{T}(σ, κ, dλ)
end

function material_response(m::GeneralPlastic, dε::SymmetricTensor{2,3,T,6}, state::GeneralPlasticState{T},
    Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

    nlsolve_cache = cache.nlsolve_cache

    σ_trial = state.σ + m.Celas ⊡ dε

    # εᵖ_equi = get_equivalent_Hill(state.εᵖ, m)
    Φ = m.yieldFunction(σ_trial) - m.yieldStress(state.κ)

    if Φ <= 0
        return σ_trial, m.Celas, GeneralPlasticState(state.εᵖ, state.εᵉ+dε, σ_trial, state.κ)
    else
        # set the current residual function that depends only on the variables
        # nlsolve_cache.f = (r_vector, x_vector) -> vector_residual!(((x)->residuals(x,m,state,Δε)), r_vector, x_vector, m)
        f(r_vector, x_vector) = vector_residual!(((x)->residuals(x,m,state,dε)), r_vector, x_vector, m)
        update_cache!(nlsolve_cache, f)
        # initial guess
        x0 = ResidualsGeneralPlastic(σ_trial, state.κ, 0.0)
        # convert initial guess to vector
        tomandel!(nlsolve_cache.x_f, x0)
        # solve for variables x
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
        result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)
        println("norm(ε) = ", string(norm(state.εᵉ+ state.εᵖ))," iterations: ", string(result.iterations))
        
        if result.f_converged
            x = frommandel(ResidualsGeneralPlastic, result.zero::Vector{T})
            df_dσ = Tensors.gradient(m.yieldFunction, x.σ)
            dεᵖ = x.dλ*df_dσ
            dεᵉ = dε - dεᵖ
            εᵖ = state.εᵖ + dεᵖ
            Cep = m.Celas # it must be corrected later!
            return x.σ, Cep, GeneralPlasticState(εᵖ, state.εᵉ+dεᵉ, x.σ, x.κ)
        else
            error("Material model not converged. Could not find material state.")
        end

    end

end

function residuals(vars::ResidualsGeneralPlastic, m::GeneralPlastic, state::GeneralPlasticState, dε)

    df_dσ = Tensors.gradient(m.yieldFunction, vars.σ)
    dεᵖ = vars.dλ * df_dσ 
    # εᵖ = state.εᵖ + dεᵖ
    Rσ = vars.σ - state.σ + m.Celas ⊡ (dεᵖ - dε)
    Rκ = vars.κ - state.κ - vars.dλ
    # εᵖ_equi = get_equivalent_Hill(εᵖ, m)
    RΦ = m.yieldFunction(vars.σ) - m.yieldStress(vars.κ)

    return ResidualsGeneralPlastic(Rσ, Rκ, RΦ)
end