# Regenerates test/fixtures/*.jld2. Run manually in any env with
# TimeseriesBase + JLD2 + DimensionalData, e.g.
#   julia --project=@. test/gen_fixtures.jl
#
# The fixtures embed a custom `@dim` (Foo/Bar) and a foreign metadata type (Meta)
# defined in module `FixtureGen`, which the test process does NOT load. On read
# those types are therefore absent, which is exactly the graceful-fallback path
# the JLD2 extension must handle (a missing custom dim type must degrade to a
# generic `Dim{name}` carrying the same lookup, not poison the whole array).
using TimeseriesBase, JLD2, DimensionalData

module FixtureGen
    using DimensionalData
    DimensionalData.@dim Foo "Foo (absent on read)"
    DimensionalData.@dim Bar "Bar (absent on read)"
    struct Meta
        note::String
    end
end

dir = joinpath(@__DIR__, "fixtures")
mkpath(dir)

# 1. ToolsArray: custom Foo dim (Categorical) + 𝑡 dim (Regular Sampled), Dict metadata.
x = ToolsArray(
    collect(reshape(1.0:12.0, 3, 4)),
    (FixtureGen.Foo([:a, :b, :c]), 𝑡(0.1:0.1:0.4));
    metadata = Dict(:k => 1),
)
jldsave(joinpath(dir, "absent_toolsarray.jld2"); x)

# 2. ToolsArray with a foreign metadata type (must not cascade).
xm = ToolsArray(collect(1.0:3.0), (FixtureGen.Foo([:a, :b, :c]),); metadata = FixtureGen.Meta("hi"))
jldsave(joinpath(dir, "absent_foreignmeta.jld2"); x = xm)

# 3. DimArray outer grid (custom Foo,Bar dims) of ToolsArray cells (𝑡 dim).
cells = [ToolsArray(collect(1.0:2.0) .+ (10i + j), (𝑡(0.1:0.1:0.2),)) for i in 1:2, j in 1:3]
grid = DimArray(cells, (FixtureGen.Foo([:p, :q]), FixtureGen.Bar([:x, :y, :z])))
jldsave(joinpath(dir, "absent_dimarray_grid.jld2"); x = grid)

println("wrote fixtures to ", dir)
