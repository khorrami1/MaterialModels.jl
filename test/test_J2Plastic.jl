
@testset begin

    # using Plots

    function uniaxialTest(loadingRange, Δε)

        m = J2Plasticity(;E=210e3, ν=0.3, σ0=200.0, H=10e-1)
        cache = get_cache(m)
        state = initial_material_state(m)
        e11_all = Float64[]
        s11_all = Float64[]

        push!(e11_all, 0.0)
        push!(s11_all, 0.0)

        ε = zero(SymmetricTensor{2,3,Float64})

        for e11 in loadingRange
            # Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)
            ε += Δε
            σ, ∂σ∂ε, state = material_response(m, ε, state, nothing, cache, nothing)
            push!(e11_all, e11)
            push!(s11_all, σ[1,1])
        end
        return e11_all, s11_all, state
    end

    loadingRange = range(0.0, 0.002, 201)
    Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : 0.0)

    e11_all, s11_all, state = uniaxialTest(loadingRange, Δε)


    # p = plot(e11_all, s11_all)


    loadingRange = range(0.0, 0.002, 201)
    Δε = SymmetricTensor{2,3,Float64}((i,j) -> i==1 && j==1 ? loadingRange.step.hi : (i==2 && j==2 ? loadingRange.step.hi : 0.0))

    e11_all, s11_all, state = uniaxialTest(loadingRange, Δε)
    # p = plot(e11_all, s11_all)

end