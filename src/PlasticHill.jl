
struct mat_Hill48{T}<:MMB.AbstractMaterial
    YS :: T #Yiels stress (MPa)
    UTS :: T #Ultimate Tensile strength (MPa)
    E :: T #Elastic Modulus (MPa)
    R0 :: T
    R45 :: T
    R90 :: T
    K :: T # strength index (coefficient in hardening law)
    n :: T #Strain hardening exponent
end

mat_AA2024_O = mat_Hill48(72.0, 121.0, 73100.0, 0.65, 0.83, 0.6, 326.8, 0.226)

function get_coefs_Hill48(m::mat_Hill48)
    R0 = m.R0
    R45 = m.R45
    R90 = m.R90
    F = R0/(R90*(R0+1))
    G = 1/(1+R0)
    H = R0/(1+R0)
    N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
    return F, G, H, N
end

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    N::T
end 

function (f::Yield_Hill48)(σ::SymmetricTensor{2})
    return 0.5*(f.F*(σ[2,2]-σ[3,3])*(σ[2,2]-σ[3,3]) + 
                f.G*(σ[3,3]-σ[1,1])*(σ[3,3]-σ[1,1]) +
                f.H*(σ[1,1]-σ[2,2])*(σ[1,1]-σ[2,2]))+f.N*σ[1,2]*σ[1,2]-0.5
end 

coefs_Hill = get_coefs_Hill48(mat_AA2024_O)

yield_func = Yield_Hill48(coefs_Hill...)

yield_func(s)

Tensors.gradient(yield_func, s)

# ̇εₚ = ̇λ*∂f∂σ, where λ=̄εₚ (equivalnet plastic strain), Hardening Law: Y = K*εⁿ (yield stress!)
