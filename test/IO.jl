@testitem "IO" tags=[:fast] begin
    import TimeseriesBase: Timeseries
    using Unitful
    using JLD2
    x = Timeseries(rand(1000, 3), 0.001:0.001:1, 1:3; metadata = Dict(:a => :test),
                   name = "name")

    f = tempname() * ".jld2"
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test x == _x

    f = tempname() * ".tsv"
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test all(x .≈ _x)

    x = x[:, 1]
    savetimeseries(f, x)
    _x = @test_logs (:warn, "Cannot load refdims yet") loadtimeseries(f)
    @test refdims(_x) == ()
    @test all(x .≈ _x)

    x = Timeseries(rand(1000, 3), 0.001:0.001:1, 1:3; metadata = Dict(:a => :test))
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test x ≈ _x

    x = Timeseries(rand(1000, 3), 0.001:0.001:1, X(1:3); metadata = Dict(:a => :test))
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test x ≈ _x
    @test [all(d .≈ _d) for (d, _d) in zip(dims(x), dims(_x))] |> all
    @test parent(lookup(_x, 1)) isa Vector{Float64}

    # Currently not the greatest way of handling non-serializable metadata
    x = Timeseries(rand(1000, 3), 0.001:0.001:1, 1:3;
                   metadata = Dict(:a => DimensionalData.NoName())) # Something that can't be serialized
    savetimeseries(f, x)
    # @test_logs (:warn, r"Cannot serialize type") savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test Dict(_x.metadata) == Dict{String, Any}("a" => "")
    # @test metadata(_x) == DimensionalData.Dimensions.NoMetadata()
    @test all(x .≈ _x)
    @test [all(d .≈ _d) for (d, _d) in zip(dims(x), dims(_x))] |> all
    @test parent(lookup(_x, 1)) isa Vector{Float64}

    x = Timeseries(rand(1000, 3), 0.001:0.001:1, 1:3; name = :Timeseries)
    savetimeseries(f, x)
    # @test_logs (:warn, r"Cannot serialize type") savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test name(_x) == "Timeseries"
    @test all(x .≈ _x)
    @test [all(d .≈ _d) for (d, _d) in zip(dims(x), dims(_x))] |> all
    @test parent(lookup(_x, 1)) isa Vector{Float64}

    x = Timeseries(rand(1000, 3), 0.001:0.001:1, [Timeseries, Timeseries, Timeseries])
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test all(parent(x) .≈ parent(_x))
    @test parent(lookup(_x, 1)) isa Vector{Float64}
    @test parent(lookup(_x, 2)) isa Vector{Symbol}

    x = Timeseries(rand(1000), 0.001:0.001:1)
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test all(x .≈ _x)
    @test [all(d .≈ _d) for (d, _d) in zip(dims(x), dims(_x))] |> all
    @test parent(lookup(_x, 1)) isa Vector{Float64}

    x = Timeseries(rand(1000, 3), (0.001:0.001:1) * u"s", 1:3; metadata = Dict(:a => :test),
                   name = "name") * u"V"

    f = tempname() * ".jld2"
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test x == _x
end
