@testitem "IO" tags = [:fast] begin
    import TimeseriesBase: Timeseries
    using Unitful
    using JLD2
    x = Timeseries(
        rand(1000, 3), 0.001:0.001:1, 1:3; metadata = Dict(:a => :test),
        name = "name"
    )

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
    x = Timeseries(
        rand(1000, 3), 0.001:0.001:1, 1:3;
        metadata = Dict(:a => DimensionalData.NoName())
    ) # Something that can't be serialized
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

    x = Timeseries(
        rand(1000, 3), (0.001:0.001:1) * u"s", 1:3; metadata = Dict(:a => :test),
        name = "name"
    ) * u"V"

    f = tempname() * ".jld2"
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test x == _x
end

@testitem "IO: unsupported TSV load fails loudly" tags = [:fast] begin
    # A TSV whose first line is not the '#' metadata header is the (unsupported)
    # flattened DimTable layout written for ≥3D series. Loading must throw a clear
    # error rather than silently returning `nothing`.
    f = tempname() * ".tsv"
    write(f, "a\tb\n1\t2\n")
    @test_throws ArgumentError loadtimeseries(f)
end

@testitem "IO: TSV round trip preserves units" tags = [:fast] begin
    using Unitful
    x = Timeseries(collect(1.0:5.0) * u"V", (1.0:5.0) * u"s")
    f = tempname() * ".tsv"
    savetimeseries(f, x)
    _x = loadtimeseries(f)
    @test eltype(_x) == eltype(x)
    @test all(parent(_x) .≈ parent(x))
    @test unit(eltype(times(_x))) == u"s"
    @test all(ustripall(times(_x)) .≈ ustripall(times(x)))
end

@testitem "IO: empty TSV fails loudly" tags = [:fast] begin
    f = tempname() * ".tsv"
    write(f, "")
    @test_throws ArgumentError loadtimeseries(f)
end

@testitem "IO: TSV column header for a univariate slice" tags = [:fast] begin
    X = Timeseries(randn(6, 2), 1.0:6.0, [:a, :b])
    u = X[Var(At(:a))]
    @test length(refdims(u)) == 1
    f = tempname() * ".tsv"
    savetimeseries(f, u)
    header = split(read(f, String), '\n')[5]
    @test header == "𝑡\ta"        # the refdim's value, not the dimension's type
    @test !occursin("{", header)
    _u = @test_logs (:warn, "Cannot load refdims yet") loadtimeseries(f)
    @test all(parent(_u) .≈ parent(u))
end

@testitem "Dates: TSV round trip" tags = [:fast] begin
    using Dates, Unitful
    dir = mktempdir()
    # The lookup comes back as a vector, not a range, as it does for every TSV load.
    function roundtrip(nm, x)
        f = joinpath(dir, "$nm.tsv")
        savetimeseries(f, x)
        return loadtimeseries(f)
    end

    t = DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 10)
    x = Timeseries(randn(10), t)
    y = roundtrip("dt", x)
    @test eltype(times(y)) == DateTime
    @test times(y) == collect(t)
    # Once the time column holds strings, `readdlm` returns a `Matrix{Any}`; the data
    # must not come back as `Vector{Any}`.
    @test eltype(parent(y)) == Float64
    @test parent(y) ≈ parent(x)

    d = Timeseries(randn(10), Date(2020, 1, 1):Day(1):Date(2020, 1, 10))
    yd = roundtrip("date", d)
    @test eltype(times(yd)) == Date
    @test times(yd) == collect(times(d))

    # `writedlm` prints a Period as "1 day", which nothing parses, so periods travel
    # as bare counts.
    p = Timeseries(randn(10), Day(1):Day(1):Day(10))
    yp = roundtrip("period", p)
    @test eltype(times(yp)) == Day
    @test times(yp) == collect(times(p))

    X = Timeseries(randn(10, 3), t, [:a, :b, :c])
    yX = roundtrip("multi", X)
    @test eltype(times(yX)) == DateTime
    @test eltype(parent(yX)) == Float64
    @test parent(yX) ≈ parent(X)

    # Metadata still survives alongside the reserved key.
    ym = roundtrip("md", Timeseries(randn(10), t; metadata = Dict(:foo => 1)))
    @test metadata(ym)["foo"] == 1
    @test eltype(times(ym)) == DateTime

    # A lookup type the load path cannot rebuild is refused on save, not on load.
    @test_throws ArgumentError savetimeseries(
        joinpath(dir, "time.tsv"),
        Timeseries(randn(3), [Time(0), Time(1), Time(2)])
    )

    # The numeric and Unitful paths are untouched.
    yn = roundtrip("num", Timeseries(randn(10), 1.0:1.0:10.0))
    @test eltype(times(yn)) == Float64
    yu = roundtrip("u", Timeseries(randn(10)u"V", (1.0:1.0:10.0)u"s"))
    @test eltype(times(yu)) == typeof(1.0u"s")
    @test eltype(parent(yu)) == typeof(1.0u"V")
end
