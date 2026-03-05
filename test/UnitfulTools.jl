
@testitem "Unitful" tags=[:fast] begin
    using Unitful
    ts = (1:1000)u"s"
    x = @test_nowarn Timeseries(randn(1000), ts)
    @test Timeseries(collect(x), ustripall(ts)u"s") == x
    @test x isa AbstractTimeseries
    @test x isa UnitfulTimeseries
    @test x isa RegularTimeseries
    @test x isa UnivariateTimeseries

    @test step(x) == step(ts)
    @test samplingrate(x) == 1 / step(ts)
    @test times(x) == ts
    @test duration(x) == -first(-(extrema(ts)...))
    @test x[𝑡(1u"s" .. 10u"s")] == x[1:10]
    @test x[𝑡 = 1:10] == x[1:10]

    @test_nowarn rectify(x; dims = 𝑡)
end
