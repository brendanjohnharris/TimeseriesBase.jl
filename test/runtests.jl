using TestItems
using TestItemRunner

@run_package_tests

@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(TimeseriesBase, unbound_args = true)
end

@testitem "JET" begin
    if isempty(VERSION.prerelease)
        import Pkg
        Pkg.add("JET")
        using JET
        mods = (TimeseriesBase,)
        JET.test_package(TimeseriesBase; target_modules = mods)

        rts = Timeseries(randn(1000), 0.01:0.01:10.0)
        θ = rand(100)
        TF = typeof(rts)
        VF = typeof(θ)
        JET.test_opt(Tuple{typeof(times), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(step), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(samplingrate), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(samplingperiod), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(nyquist), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(duration), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(centraldiff), TF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(phasegrad), Float64, Float64}; target_modules = mods)
        JET.test_opt(Tuple{typeof(resultant), VF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(resultantlength), VF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(circularmean), VF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(circularvar), VF}; target_modules = mods)
        JET.test_opt(Tuple{typeof(circularstd), VF}; target_modules = mods)

        if VERSION >= v"1.11"
            JET.test_opt(Tuple{typeof(centralderiv), TF}; broken = true, target_modules = mods)
        end
    else
        @warn "JET will fail on unreleased Julia versions; skipping JET tests" version = VERSION
    end
end

@testitem "Dates" tags = [:fast] begin
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

@testitem "Spectra" tags = [:fast] begin
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
