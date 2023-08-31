
# With associative follow rule
# Reference: Forming Limit Prediction of Anisotropic Aluminum Magnesium Alloy Sheet AA5052-H32 Using Micromechanical Damage Model

struct PlasticHill<:AbstractMaterial
    UTS :: Float64 #Ultimate Tensile strength (MPa)
    E :: Float64 #Elastic Modulus (MPa)
    ν :: Float64 #Poinson's ration
    R0 :: Float64
    R45 :: Float64
    R90 :: Float64

    # precomputed Hill coefficients
    F :: Float64
    G :: Float64
    H :: Float64
    N :: Float64
    Eᵉ :: SymmetricTensor{4,3,Float64,36}

    function PlasticHill(UTS, E, ν, R0, R45, R90)
        F = R0/(R90*(R0+1))
        G = 1/(1+R0)
        H = R0/(1+R0)
        N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
        Eᵉ = elastic_tangent_3D(E, ν)
        return new(UTS, E, ν, R0, R45, R90, F, G, H, N, Eᵉ)
    end
end

# keyword argument constructor
PlasticHill(;UTS, E, ν, R0, R45, R90) = PlasticHill(UTS, E, ν, R0, R45, R90)

# plasticHill = PlasticHill(72.0, 121.0, 73100.0, 0.3, 0.65, 0.83, 0.6, 326.8, 0.226)
plasticHill = PlasticHill(243.0, 69000.0, 0.33, 0.84, 0.64, 1.51)
yieldStress_fun(ϵ) = 376.9*(0.0059+ϵ)^0.152 # Swift hardening rule
# plot(ϵ->376.9*(0.0059+ϵ)^0.152, xlim=(0,0.1))

struct PlasticHillState{dim,T, M} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,dim,T,M}
    λ :: T # if associative follow rule, then Equivalent plastic strain = plastic multiplier 
    # σY :: SymmetricTensor{2,dim,T,M}
end

Base.zero(::Type{PlasticHillState{dim,T,M}}) where {dim,T,M} = PlasticHillState(zero(SymmetricTensor{2,dim,T,M}), 0.)
initial_material_state(::PlasticHill) = zero(PlasticHillState{3,Float64, 6})

struct PlasticHillCache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cache::T
end

get_n_scalar_equations(::PlasticHill) = 7

struct ResidualsPlasticHill{T}
    σ::SymmetricTensor{2,3,T,6}
    λ::T
end

function get_cache(m::PlasticHill)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return PlasticHillCache(cache)
end


struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    N::T
end 

function (f::Yield_Hill48)(σ::SymmetricTensor{2,3})
    return sqrt(1.5*( f.H*(σ[1,1]-σ[2,2])*(σ[1,1]-σ[2,2]) + f.F*(σ[2,2]-σ[3,3])*(σ[2,2]-σ[3,3]) + f.G*(σ[3,3]-σ[1,1])*(σ[3,3]-σ[1,1]) + 2*f.N*( σ[1,2]*σ[1,2] + σ[2,3]*σ[2,3] + σ[1,3]*σ[1,3] ) )/(f.F+f.G+f.H))
end 


yield_func = Yield_Hill48(plasticHill.F, plasticHill.G, plasticHill.H, plasticHill.N)

function Tensors.tomandel!(v::Vector{T}, r::ResidualsPlasticHill{T}) where T
    M=6
    # TODO check vector length
    tomandel!(view(v, 1:M), r.σ)
    v[M+1] = r.λ
    return v
end

function Tensors.frommandel(::Type{ResidualsPlasticHill}, v::Vector{T}) where T
    σ = frommandel(SymmetricTensor{2,3}, view(v, 1:6))
    λ = v[7]
    return ResidualsPlasticHill{T}(σ, λ)
end

function material_response(m::PlasticHill, ε::SymmetricTensor{2,3,T,6}, state::PlasticHillState{3},
    Δt=nothing; cache=get_cache(m), options::Dict{Symbol, Any} = Dict{Symbol, Any}()) where T

    nlsolve_cache = cache.nlsolve_cache

    σ_trial = m.Eᵉ ⊡ (ε - state.εᵖ)

    Φ = yield_func(σ_trial) - yieldStress_fun(state.λ)

    if Φ <= 0
        return σ_trial, m.Eᵉ, state
    else
        # set the current residual function that depends only on the variables
        # nlsolve_cache.f = (r_vector, x_vector) -> vector_residual!(((x)->residuals(x,m,state,Δε)), r_vector, x_vector, m)
        f(r_vector, x_vector) = vector_residual!(((x)->residuals(x,m,state,ε)), r_vector, x_vector, m)
        update_cache!(nlsolve_cache, f)
        # initial guess
        x0 = ResidualsPlasticHill(σ_trial,state.λ)
        # convert initial guess to vector
        tomandel!(nlsolve_cache.x_f, x0)
        # solve for variables x
        nlsolve_options = get(options, :nlsolve_params, Dict{Symbol, Any}(:method=>:newton))
        haskey(nlsolve_options, :method) || merge!(nlsolve_options, Dict{Symbol, Any}(:method=>:newton)) # set newton if the user did not supply another method
        result = NLsolve.nlsolve(nlsolve_cache, nlsolve_cache.x_f; nlsolve_options...)
        
        if result.f_converged
            x = frommandel(ResidualsPlasticHill, result.zero::Vector{T})
            εᵖ = state.εᵖ + (x.λ - state.λ)*Tensors.gradient(yield_func, x.σ)
            Cep = m.Eᵉ # it must be corrected later!
            return x.σ, Cep, PlasticHillState(εᵖ, x.λ)
        else
            error("Material model not converged. Could not find material state.")
        end

    end

end


function residuals(vars::ResidualsPlasticHill, m::PlasticHill, material_state::PlasticStateHill, dε)

    Rσ = vars.σ - material_state.
    yield_func()

    return ResidualsPlastic(Rσ, Rκ, Rα, Rμ)
end

stress_test = zero(SymmetricTensor{2,3})

yield_func(stress_test)

df_dσ = Tensors.gradient(yield_func, stress_test)

# ̇εₚ = ̇λ*∂f∂σ, where λ=̄εₚ (equivalnet plastic strain), Hardening Law: Y = K*εⁿ (yield stress!)
# λ = sqrt(2/3*dev(εₚ)⊡dev(εₚ))

