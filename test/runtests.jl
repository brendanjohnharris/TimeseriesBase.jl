using TestItems
using TestItemRunner

@run_package_tests

@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(TimeseriesBase, unbound_args = true)
end

@testitem "Dates" begin
    using Dates, Unitful
    x = 1:100
    t = DateTime(1901):Year(1):DateTime(2000)
    y = @test_nowarn Timeseries(x, t)
    @test y isa RegularTimeseries
    @test samplingperiod(y) == Year(1)
    @test times(y) == t
    @test duration(y) == last(t) - first(t)
    @test unit(y) == NoUnits
end

@testitem "Spectra" begin
    using Unitful
    # Define a test time series
    fs = 0.1:0.1:100
    S = 1.0 ./ fs
    Pxx = Spectrum(fs, S)
    @test Pxx isa RegularSpectrum
    # ........and other funcs
end

include("ToolsArrays.jl")
include("Utils.jl")
include("UnitfulTools.jl")
include("Operators.jl")
include("SciMLBaseExt.jl")
