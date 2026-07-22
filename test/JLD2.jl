@testitem "JLD2 typemap matcher" tags = [:fast] begin
    using JLD2
    tm = TimeseriesBase.toolsarray_typemap
    # ToolsArray paths (current + legacy TimeseriesTools) and plain DimArray → Upgrade
    @test tm(nothing, "TimeseriesBase.ToolsArrays.ToolsArray", []) isa JLD2.Upgrade
    @test tm(nothing, "TimeseriesBase.ToolsArrays.ToolsArray", []).target === ToolsArray
    @test tm(nothing, "TimeseriesTools.ToolsArray", []) isa JLD2.Upgrade          # legacy files (G5)
    @test tm(nothing, "DimensionalData.DimArray", []) isa JLD2.Upgrade
    @test tm(nothing, "DimensionalData.DimArray", []).target === DimArray
end

@testitem "JLD2 round-trip, types present (G2/R2/R4)" tags = [:fast] begin
    using JLD2, DimensionalData
    f = tempname() * ".jld2"

    # G2: type-identical round trip (Var/𝑡 are TB-owned, always present)
    x = ToolsArray(
        collect(reshape(1.0:12.0, 3, 4)), (Var([:a, :b, :c]), 𝑡(0.1:0.1:0.4));
        metadata = Dict(:k => 1),
    )
    jldsave(f; x)
    y = loadtoolsarray(f, "x")
    @test y == x
    @test typeof(y) == typeof(x)
    @test dims(y) == dims(x)
    @test metadata(y) == metadata(x)

    # R2: Regular sampling preserved, not re-inferred to Irregular
    @test DimensionalData.span(lookup(dims(y, 2))) isa DimensionalData.Regular

    # same result through the jldopen do-block API using the exported typemap
    y2 = jldopen(f; typemap = toolsarray_typemap) do file
        file["x"]
    end
    @test y2 == x
    @test typeof(y2) == typeof(x)

    # R4: nested ToolsArray-of-ToolsArray
    cells = [ToolsArray(collect(1.0:2.0) .+ (10i + j), (𝑡(0.1:0.1:0.2),)) for i in 1:2, j in 1:3]
    outer = ToolsArray(cells, (Var([:p, :q]), 𝑡(1.0:1.0:3.0)))
    jldsave(f; x = outer)
    yo = loadtoolsarray(f, "x")
    @test typeof(yo) == typeof(outer)
    @test yo[1, 1] isa AbstractToolsArray
    @test yo[1, 1] == outer[1, 1]

    # edges: refdims, and a 0-dim array
    xr = ToolsArray(
        collect(reshape(1.0:6.0, 2, 3)), (Var([:a, :b]), 𝑡(0.1:0.1:0.3));
        refdims = (Obs(1),),
    )
    jldsave(f; x = xr)
    yr = loadtoolsarray(f, "x")
    @test typeof(yr) == typeof(xr)
    @test refdims(yr) == refdims(xr)

    x0 = ToolsArray(fill(3.0), ())
    jldsave(f; x = x0)
    y0 = loadtoolsarray(f, "x")
    @test parent(y0)[] == 3.0
end

@testitem "JLD2 robust load, custom types absent (G1/G3 + DimArray)" tags = [:fast] begin
    using JLD2, DimensionalData
    fixt(n) = joinpath(@__DIR__, "fixtures", n)

    # G1: absent custom dim → generic Dim{:Foo}, exact lookup, usable, loud warning
    y = @test_logs (:warn,) match_mode = :any loadtoolsarray(fixt("absent_toolsarray.jld2"), "x")
    @test y isa ToolsArray
    @test name(dims(y, 1)) == :Foo
    @test dims(y, 1) isa DimensionalData.Dim               # fell back to generic
    @test collect(lookup(dims(y, 1))) == [:a, :b, :c]      # lookup preserved exactly
    @test name(dims(y, 2)) == :𝑡                           # TB-owned → real 𝑡, not generic
    @test !(dims(y, 2) isa DimensionalData.Dim)
    @test DimensionalData.span(lookup(dims(y, 2))) isa DimensionalData.Regular   # R2
    @test parent(y) == collect(reshape(1.0:12.0, 3, 4))
    @test collect(y[Dim{:Foo}(At(:b))]) == [2.0, 5.0, 8.0, 11.0]   # selectable by name

    # G3: foreign metadata does not cascade — array still usable, data intact
    ym = loadtoolsarray(fixt("absent_foreignmeta.jld2"), "x")
    @test ym isa ToolsArray
    @test parent(ym) == [1.0, 2.0, 3.0]
    @test !(metadata(ym) isa Dict)                         # opaque but present, not fatal

    # DimArray outer grid → rebuilt as DimArray (not ToolsArray), cells intact ToolsArrays
    yg = loadtoolsarray(fixt("absent_dimarray_grid.jld2"), "x")
    @test yg isa DimArray
    @test !(yg isa AbstractToolsArray)
    @test name.(dims(yg)) == (:Foo, :Bar)
    @test yg[1, 1] isa AbstractToolsArray
    @test parent(yg[1, 1]) == [12.0, 13.0]                 # cell i=1,j=1: (1:2) .+ 11
end
