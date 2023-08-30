
struct PlasticHill<:AbstractMaterial
    YS :: Float64 #Yiels stress (MPa)
    UTS :: Float64 #Ultimate Tensile strength (MPa)
    E :: Float64 #Elastic Modulus (MPa)
    ν :: Float64 #Poinson's ration
    R0 :: Float64
    R45 :: Float64
    R90 :: Float64
    K :: Float64 # strength index (coefficient in hardening law)
    n :: Float64 #Strain hardening exponent

    # precomputed Hill coefficients
    F :: Float64
    G :: Float64
    H :: Float64
    N :: Float64
    Eᵉ :: SymmetricTensor{4,3,Float64,36}

    function PlasticHill(YS, UTS, E, ν, R0, R45, R90, K, n)
        F = R0/(R90*(R0+1))
        G = 1/(1+R0)
        H = R0/(1+R0)
        N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
        Eᵉ = elastic_tangent_3D(E, ν)
        return new(YS, UTS, E, ν, R0, R45, R90, K, n, F, G, H, N, Eᵉ)
    end
end

# keyword argument constructor
PlasticHill(; YS, UTS, E, ν, R0, R45, R90, K, n) = PlasticHill(YS, UTS, E, ν, R0, R45, R90, K, n)

plasticHill = PlasticHill(72.0, 121.0, 73100.0, 0.3, 0.65, 0.83, 0.6, 326.8, 0.226)


struct PlasticHillState{dim,T, M} <: AbstractMaterialState
    εᵖ::SymmetricTensor{2,dim,T,M}
    λdot :: T
    σY :: SymmetricTensor{2,dim,T,M}
end

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    N::T
end 

function (f::Yield_Hill48)(σ::SymmetricTensor{2,3})
    return 0.5*(f.F*(σ[2,2]-σ[3,3])*(σ[2,2]-σ[3,3]) + 
                f.G*(σ[3,3]-σ[1,1])*(σ[3,3]-σ[1,1]) +
                f.H*(σ[1,1]-σ[2,2])*(σ[1,1]-σ[2,2]))+f.N*σ[1,2]*σ[1,2]-0.5
end 


yield_func = Yield_Hill48(plasticHill.F, plasticHill.G, plasticHill.H, plasticHill.N)

stress_test = zero(SymmetricTensor{2,3})

yield_func(stress_test)

df_dσ = Tensors.gradient(yield_func, stress_test)

# ̇εₚ = ̇λ*∂f∂σ, where λ=̄εₚ (equivalnet plastic strain), Hardening Law: Y = K*εⁿ (yield stress!)
# λ = sqrt(2/3*dev(εₚ)⊡dev(εₚ))