@testitem "SciMLBaseExt" begin
    using SciMLBase
    using DiffEqBase
    using OrdinaryDiffEqTsit5

    sol = SciMLBase.build_solution(ODEProblem((u, p, t) -> u, 1.0, (0.0, 1.0)),
                                   :NoAlgorithm, 0.0:0.1:1.0, exp.(0.0:0.1:1.0))

    @test sol isa SciMLBase.AbstractTimeseriesSolution

    x = @test_nowarn Timeseries(sol)
    @test x isa TimeseriesBase.UnivariateRegular
    @test parent(x) ≈ exp.(0.0:0.1:1.0)

    # * Multivariate
    f = (u, p, t) -> [u, u]
    sol = SciMLBase.build_solution(ODEProblem(f, 1.0, (0.0, 1.0)),
                                   :NoAlgorithm, 0.0:0.1:1.0, f.(0.0:0.1:1.0, 0, 0))
    x = Timeseries(sol)

    @test x isa TimeseriesBase.MultivariateRegular

    # * Generic array solution
    f = (u, p, t) -> [u u; u u]
    sol = SciMLBase.build_solution(ODEProblem(f, 1.0, (0.0, 1.0)),
                                   :NoAlgorithm, 0.0:0.1:1.0, f.(0.0:0.1:1.0, 0, 0))
    x = Timeseries(sol)

    @test x isa RegularTimeseries
    @test first(x) == first(sol.u)

    A = [1 2
         3 4]
    prob = ODEProblem((u, p, t) -> A * u, ones(2, 2), (0.0, 1.0))
    function prob_func(prob, i, repeat)
        remake(prob, u0 = i * prob.u0)
    end
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
    sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 10, saveat = 0.01)
    @test sol isa EnsembleSolution

    x = Timeseries(sol)
    @test x isa ToolsArray{<:AbstractToolsArray, 1}
    @test dims(x)[1] == Obs(1:10)
end
