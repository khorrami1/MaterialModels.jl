
using MaterialModels
using Tensors
using Plots


yieldStress(ϵ) = 376.9*(0.0059+ϵ)^0.152
E = 69e3
ν = 0.3
Celas = elastic_tangent_3D(E, ν)

struct Yield_Hill48{T}
    F::T
    G::T
    H::T
    M::T
    N::T
    L::T
end 

function (f::Yield_Hill48)(T::SymmetricTensor{2,3})
    return sqrt( (f.F*T[1,1]*T[1,1] + f.G*T[2,2]*T[2,2] + f.H*T[3,3]*T[3,3])/(f.F*f.G + f.F*f.H + f.G*f.H) + 2*T[2,3]*T[2,3]/f.L + 2*T[3,1]*T[3,1]/f.M + 2*T[1,2]*T[1,2]/f.N )
end

R0 =  0.84
R45 = 0.64
R90 = 1.51

F = R0/(R90*(R0+1))
G = 1/(1+R0)
H = R0/(1+R0)
N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
L = N # must be checked!
M = N # must be checked!

yield_Hill48 = Yield_Hill48(F, G, H, M, N, L)
yieldFunction1(T::SymmetricTensor{2,3}) = yield_Hill48(T)


m = GeneralPlastic(Celas, yieldStress, yieldFunction1)


function uniaxialTest(m, loadingRange, Δε)
    cache = get_cache(m)
    state = initial_material_state(m)
    e_all = Float64[]
    s_all = Float64[]

    for e11 in loadingRange
        # Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)
        σ, ∂σ∂ε, state = material_response(m, Δε, state; cache=cache)
        push!(e_all, e11)
        push!(s_all, σ[1,1])
    end
    return e_all, s_all, state
end

loadingRange = range(0.0, 0.2, 201)
Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)

e_all, s_all, state = uniaxialTest(m, loadingRange, Δε)
p = plot(e_all, s_all)
