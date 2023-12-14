

struct GTN{T} <: AbstractMaterial
    Celas :: SymmetricTensor{4, 3, T, 36}
    yieldFunction :: Function
end

struct GTNstate{T}
    εᵖ::SymmetricTensor{2,3,T} # total plastic strain
    εᵉ::SymmetricTensor{2,3,T} # total elastic strain
    σ :: SymmetricTensor{2,3,T} # Cauchy stress
    κ :: T
    f :: T # Porosity
end

initial_material_state(::GTN) = GTNstate(zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), zero(SymmetricTensor{2,3}), 0.0, 0.0)

struct GTNcache{T<:NLsolve.OnceDifferentiable} <: AbstractCache
    nlsolve_cahce::T
end

get_n_scalar_equations(::GTN) = 8

struct ResidualsGTN{T}
    σ::SymmetricTensor{2,3,T,6}
    κ::T
    dλ::T
end

Tensors.get_base(::Type{GTN{T}}) where T = ResidualsGTN # needed for frommandel

function get_cache(m::GTN)
    state = initial_material_state(m)
    f(r_vector, x_vector) = vector_residual!(((x)->MaterialModels.residuals(x, m, state, zero(SymmetricTensor{2,3}))), r_vector, x_vector, m)
    v_cache = Vector{Float64}(undef, get_n_scalar_equations(m))
    cache = NLsolve.OnceDifferentiable(f, v_cache, v_cache; autodiff = :forward)
    return GTNcache(cache)
end

function Tensors.tomandel!(v::Vector{T}, r::ResidualsGTN{T}) where T
    M=6
    # TODO check vector length
    tomandel!(view(v, 1:M), r.σ)
    v[M+1] = r.κ
    v[M+2] = r.dλ
    return v
end

function material_response(m::GTN, dε::SymmetricTensor{2,3,T,6}, state::GTNstate{T},
end