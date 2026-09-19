@testitem "Operators" tags = [:fast] begin
    import TimeseriesBase.Operators: ℬ!, ℒ!, ℬ, ℒ, 𝒯
    x = Timeseries(x -> randn(), 1:1000)
    _x = deepcopy(x)
    @test_nowarn ℬ!(x)
    @test all(x[2:end] .== parent(_x)[1:(end - 1)])
    ℬ!(x, 3)
    @test all(x[5:end] .== parent(_x)[1:(end - 4)])
    @test ℬ(_x, 4) == x

    x = deepcopy(_x)
    @test_nowarn ℒ!(x)
    @test all(_x[2:end] .== parent(x)[1:(end - 1)])
    ℒ!(x, 3)
    @test all(_x[5:end] .== parent(x)[1:(end - 4)])
    @test ℒ(_x, 4) == x

    x = Timeseries(x -> randn(), 0.0:0.01:1)
    T = 𝒯(-1)
    @test times(T(x)) == -1:step(x):0
end

@testitem "Shift operators work on multivariate series" tags = [:fast] begin
    D = randn(6, 3)
    X = Timeseries(copy(D), 1.0:6.0, [:a, :b, :c])
    # shifts act along time (the first dimension), per column
    @test parent(ℬ(X)) == circshift(D, 1)
    @test parent(ℒ(X)) == circshift(D, -1)
    @test parent(ℬ(X, 2)) == circshift(D, 2)
    @test times(ℬ(X)) == times(X)
    @test dims(ℬ(X), Var) == dims(X, Var)
    v = Timeseries(collect(1.0:5.0), 1.0:5.0) # univariate unchanged
    @test parent(ℬ(v)) == circshift(collect(1.0:5.0), 1)
end
