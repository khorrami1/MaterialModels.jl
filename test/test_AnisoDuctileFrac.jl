

using MaterialModels
using Tensors
using Plots

# paper: https://www.sciencedirect.com/science/article/pii/S0020768319300162

# Swift, loading direction:0.0
yieldStress(ϵ) = 562.8*(0.0268+ϵ)^0.1805


E = 67e3
ν = 0.33
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

# R0 =  0.84
# R45 = 0.64
# R90 = 1.51

# F = R0/(R90*(R0+1))
# G = 1/(1+R0)
# H = R0/(1+R0)
# N = (R0+R90)*(1+2*R45)/(2*R90*(1+R0))
# L = N # must be checked!
# M = N # must be checked!

# for the material AA6082-T6
F = 0.5092
G = 0.5055 
H = 0.4945
L = 1.5
M = 1.5
N = 1.5482

yield_Hill48 = Yield_Hill48(F, G, H, M, N, L)
yieldFunction1(T::SymmetricTensor{2,3}) = yield_Hill48(T)


m = GeneralPlastic(Celas, yieldStress, yieldFunction1)

A1 = 0.3994; B1 = 3.3481; D1 = -0.2543; C0 = 1/3
A2 = 0.3223; B2 = 0.8423; D2 = -0.4670;
A3 = 0.4988; B3 = 0.9976; D3 = 0.4988;
A4 = 2.2498; B4 = 1.5468; D4 = 4.1578;

C1(t) = A1*cos(t)^4 + B1*sin(t)^2*cos(t)^2 + D1*sin(t)^4
C2(t) = A2*cos(t)^4 + B2*sin(t)^2*cos(t)^2 + D2*sin(t)^4
C3(t) = A3*cos(t)^4 + B3*sin(t)^2*cos(t)^2 + D3*sin(t)^4
C4(t) = A4*cos(t)^4 + B4*sin(t)^2*cos(t)^2 + D4*sin(t)^4

tem_damage(η, L, t) = (2/sqrt(L^2+3))^C1(t) * ((η+(3-L)/(3*sqrt(L^2+3))+C0)/(1+C0))^C2(t) * (3/4*η^2+(3-L)/(3*sqrt(L^2+3))+1/3)^C4(t)

function get_η_L_t(σ)
    
    eigVal, eigVec = eigen(σ)
    σ1 = eigVal[3]
    σ2 = eigVal[2]
    σ3 = eigVal[1]
    t = acos(eigVec[1,3]) # RD along with x direction (1)
    η = (σ1 + σ2 + σ3)/(3*yieldFunction1(σ))
    L = (2*σ2 - σ1 - σ3)/(σ1 - σ3)

    return η, L, t
end

function LoadingTest(m, loadingRange, Δε)
    cache = get_cache(m)
    state = initial_material_state(m)
    e_all = SymmetricTensor{2, 3, Float64, 6}[]
    push!(e_all, zero(SymmetricTensor{2, 3, Float64, 6}))
    s_all = SymmetricTensor{2, 3, Float64, 6}[]
    push!(s_all, zero(SymmetricTensor{2, 3, Float64, 6}))
    damageParam = [0.0]

    ϵ1_all = Float64[]
    ϵ2_all = Float64[]

    for count in eachindex(loadingRange)
        # Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)
        κ_old = state.κ
        σ, ∂σ∂ε, state = material_response(m, Δε, state; cache=cache)
        κ_new = state.κ
        η, L, t = get_η_L_t(state.σ)
        push!(damageParam, damageParam[count] + tem_damage(η, L, t) * (κ_new - κ_old))
        push!(e_all, state.εᵉ + state.εᵖ )
        push!(s_all, σ)

        eigVals = eigvals(state.εᵉ + state.εᵖ)
        push!(ϵ1_all, maximum(eigVals))
        push!(ϵ2_all, minimum(eigVals))

        if damageParam[count+1] > 1.0
            break
        end
    end
    return e_all, s_all, ϵ1_all, ϵ2_all, state, damageParam
end

loadingRange = range(0.0, 0.5, 5001)
# Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)

Δε = SymmetricTensor{2,3,Float64}([loadingRange.step.hi, loadingRange.step.hi, loadingRange.step.hi, 0., 0., 0.])


e_all, s_all, ϵ1_all, ϵ2_all, state, damageParam = LoadingTest(m, loadingRange, Δε)
p = plot([e_all[i][1,2] for i in eachindex(e_all)], [s_all[i][1,2] for i in eachindex(s_all)])

plot([e_all[i][1,1] for i in eachindex(e_all)], damageParam[1:end])

η, L, t = get_η_L_t(state.σ)


eigvals(state.εᵉ + state.εᵖ)

# To plot Forming Limit Curve (FLC)
function get_FLC(m)

    loadingRange = range(0.0, 1.0, 501)
    loadingStep = loadingRange.step.hi

    ϵ1 = Vector{Float64}[]
    ϵ2 = Vector{Float64}[]

    for c in range(0.0, 1.0, 10)

        Δε = SymmetricTensor{2,3,Float64}([loadingStep, c*loadingStep, 0.0*loadingStep, c*loadingStep, 0., c*loadingStep])
        
        e_all, s_all, ϵ1_all, ϵ2_all, state, damageParam = LoadingTest(m, loadingRange, Δε)
        @show damageParam
        push!(ϵ1, ϵ1_all)
        push!(ϵ2, ϵ2_all)
        @show get_η_L_t(state.σ)
    end

    return ϵ1, ϵ2

end



ϵ1, ϵ2 = get_FLC(m)

plot(ϵ2[1], ϵ1[1])
for i in 2:10
    plot!(ϵ2[i], ϵ1[i])
    @show i
end

