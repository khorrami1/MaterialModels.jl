
# With associative follow rule
# Reference: Forming Limit Prediction of Anisotropic Aluminum Magnesium Alloy Sheet AA5052-H32 Using Micromechanical Damage Model

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    M::T
    N::T
    L::T
end 

# function (f::Yield_Hill48)(σ::SymmetricTensor{2,3})
#     return sqrt(1.5*( f.H*(σ[1,1]-σ[2,2])*(σ[1,1]-σ[2,2]) + f.F*(σ[2,2]-σ[3,3])*(σ[2,2]-σ[3,3]) + f.G*(σ[3,3]-σ[1,1])*(σ[3,3]-σ[1,1]) + 2*f.N*( σ[1,2]*σ[1,2] + σ[2,3]*σ[2,3] + σ[1,3]*σ[1,3] ) )/(f.F+f.G+f.H))
# end 

function (f::Yield_Hill48)(T::SymmetricTensor{2,3})
    return sqrt( (f.F*T[1,1]*T[1,1] + f.G*T[2,2]*T[2,2] + f.H*T[3,3]*T[3,3])/(f.F*f.G + f.F*f.H + f.G*f.H) + 2*T[2,3]*T[2,3]/f.L + 2*T[3,1]*T[3,1]/f.M + 2*T[1,2]*T[1,2]/f.N )
end

struct PlasticHill{T}<:AbstractMaterial
    UTS :: T #Ultimate Tensile strength (MPa)
    E :: T #Elastic Modulus (MPa)
    ν :: T #Poinson's ration
    R0 :: T
    R45 :: T
    R90 :: T
    yieldStress :: Function
    yieldFunction :: Yield_Hill48

    # precomputed Hill coefficients
    F :: T
    G :: T
    H :: T
    N :: T
    L :: T
    M :: T
    Eᵉ :: SymmetricTensor{4,3,T,36}

    function PlasticHill(UTS, E, ν, R0, R45, R90, yieldStress)
        F = R0/(R90*(R0+1))
        G = 1/(1+R0)
        H = R0/(1+R0)
        N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
        L = N # must be checked!
        M = N # must be checked!
        yieldFunction = Yield_Hill48(F, G, H, M, N, L)
        Eᵉ = elastic_tangent_3D(E, ν)
        return new{typeof(R0)}(UTS, E, ν, R0, R45, R90, yieldStress, yieldFunction, F, G, H, N, L, M, Eᵉ)
    end
end

# keyword argument constructor
PlasticHill(;UTS, E, ν, R0, R45, R90, yieldStress) = PlasticHill(UTS, E, ν, R0, R45, R90, yieldStress)

# plasticHill = PlasticHill(72.0, 121.0, 73100.0, 0.3, 0.65, 0.83, 0.6, 326.8, 0.226)
# plasticHill = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51)
# yieldStress_fun(ϵ) = 376.9*(0.0059+ϵ)^0.152 # Swift hardening rule
# plot(ϵ->376.9*(0.0059+ϵ)^0.152, xlim=(0,0.1))

struct PlasticHillState{dim,T, M} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,dim,T,M}
    εᵉ::SymmetricTensor{2,dim,T,M}
    σ :: SymmetricTensor{2,dim,T,M}
    κ :: T # if associative follow rule, then Equivalent plastic strain = plastic multiplier 
end

# Base.zero(::Type{PlasticHillState{dim,T,M}}) where {dim,T,M} = PlasticHillState(zero(SymmetricTensor{2,dim,T,M}), zero(SymmetricTensor{2,dim,T,M}), 0.)
# initial_material_state(::PlasticHill) = zero(PlasticHillState{3,Float64, 6})

initial_material_state(m::PlasticHill) = PlasticHillState(zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), 0.0)

struct PlasticHillCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

get_n_scalar_equations(::PlasticHill) = 8

struct ResidualsPlasticHill{T}
    σ::SymmetricTensor{2,3,T,6}
    κ::T
    dλ::T 
end

Tensors.get_base(::Type{PlasticHill{T}}) where T = ResidualsPlasticHill # needed for frommandel

function get_cache(m::PlasticHill)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return PlasticHillCache(cache)
end

function get_equivalent_Hill(T::SymmetricTensor{2,3}, m::PlasticHill)
    return sqrt( (m.F*T[1,1]*T[1,1] + m.G*T[2,2]*T[2,2] + m.H*T[3,3]*T[3,3])/(m.F*m.G + m.F*m.H + m.G*m.H) + 2*T[2,3]*T[2,3]/m.L + 2*T[3,1]*T[3,1]/m.M + 2*T[1,2]*T[1,2]/m.N )
end


function Tensors.tomandel!(v::Vector{T}, r::ResidualsPlasticHill{T}) where T
    # TODO check vector length
    tomandel!(view(v, 1:6), r.σ)
    v[7] = r.κ
    v[8] = r.dλ
    return v
end

function Tensors.frommandel(::Type{ResidualsPlasticHill}, v::Vector{T}) where T
    σ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    κ = v[7]
    dλ = v[8]
    return ResidualsPlasticHill{T}(σ, κ, dλ)
end

function material_response(m::PlasticHill, dε::SymmetricTensor{2,3,T,6}, state::PlasticHillState{3},
    Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

    nlsolve_cache = cache.nlsolve_cache

    σ_trial = state.σ + m.Eᵉ ⊡ dε

    # εᵖ_equi = get_equivalent_Hill(state.εᵖ, m)
    Φ = m.yieldFunction(σ_trial) - m.yieldStress(state.κ)

    if Φ <= 0
        return σ_trial, m.Eᵉ, PlasticHillState(state.εᵖ, state.εᵉ+dε, σ_trial, state.κ)
    else
        # set the current residual function that depends only on the variables
        # nlsolve_cache.f = (r_vector, x_vector) -> vector_residual!(((x)->residuals(x,m,state,Δε)), r_vector, x_vector, m)
        f(r_vector, x_vector) = vector_residual!(((x)->residuals(x,m,state,dε)), r_vector, x_vector, m)
        update_cache!(nlsolve_cache, f)
        # initial guess
        x0 = ResidualsPlasticHill(σ_trial, state.κ, 0.0)
        # convert initial guess to vector
        tomandel!(nlsolve_cache.x_f, x0)
        # solve for variables x
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
        result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)
        println("norm(ε) = ", string(norm(state.εᵉ+ state.εᵖ))," iterations: ", string(result.iterations))
        if result.f_converged
            x = frommandel(ResidualsPlasticHill, result.zero::Vector{T})
            ∂f∂σ = Tensors.gradient(m.yieldFunction, x.σ)
            dεᵖ = x.dλ * ∂f∂σ
            dεᵉ = dε - dεᵖ
            εᵖ = state.εᵖ + dεᵖ
            C_f_σ = m.Celas ⊡ ∂f∂σ
            f_σ_C = ∂f∂σ ⊡ m.Celas
            H = Tensors.gradient(m.yieldStress, state.κ)
            Cep = m.Celas - (C_f_σ ⊗ f_σ_C)/(H + ∂f∂σ ⊡ C_f_σ) # it must be corrected later!
            return x.σ, Cep, PlasticHillState(εᵖ, state.εᵉ+dεᵉ, x.σ, x.κ)
        else
            error("Material model not converged. Could not find material state.")
        end

    end

end


function residuals(vars::ResidualsPlasticHill, m::PlasticHill, state::PlasticHillState, dε)

    df_dσ = Tensors.gradient(m.yieldFunction, vars.σ)
    dεᵖ = vars.dλ * df_dσ 
    # εᵖ = state.εᵖ + dεᵖ
    Rσ = vars.σ - state.σ + m.Eᵉ ⊡ (dεᵖ - dε)
    Rκ = vars.κ - state.κ - vars.dλ
    # εᵖ_equi = get_equivalent_Hill(εᵖ, m)
    RΦ = m.yieldFunction(vars.σ) - m.yieldStress(vars.κ)

    return ResidualsPlasticHill(Rσ, Rκ, RΦ)
end