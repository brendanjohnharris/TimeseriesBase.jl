using TestItems
using TestItemRunner

@run_package_tests

@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(TimeseriesBase, unbound_args = true)
end

@testitem "JET" begin
    if isempty(VERSION.prerelease)
        using JET
        # Target TimeseriesBase and all of its submodules, so static-analysis reports from
        # e.g. Utils, IO, or UnitfulTools are included (targeting only the top module need
        # not descend into submodules).
        function submodules(m::Module, acc = Module[])
            push!(acc, m)
            for n in names(m; all = true)
                isdefined(m, n) || continue
                s = getfield(m, n)
                s isa Module && s !== m && parentmodule(s) === m && submodules(s, acc)
            end
            return acc
        end
        rep = JET.report_package(
            TimeseriesBase;
            target_modules = Tuple(unique(submodules(TimeseriesBase)))
        )
        @test isempty(JET.get_reports(rep))
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
