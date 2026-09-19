module IO

import FileIO: save, load, @format_str, File, query
using JSON
import JSON.json
using DelimitedFiles
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.Utils
using TimeseriesBase.UnitfulTools
import TimeseriesBase.UnitfulTools: _unit
import TimeseriesBase.DatesTools: _diffunit
using DimensionalData
import DimensionalData: Dates
import Unitful: NoUnits, uparse

export savetimeseries, savets, loadtimeseries, loadts, loadtoolsarray, toolsarray_typemap

"""
    loadtoolsarray(file, key)

Load `key` from a JLD2 `file`, reconstructing any stored `ToolsArray`/`DimArray`
robustly: a dimension whose custom type is absent from the current workspace comes
back as a generic `Dim{name}` carrying the original lookup (with a warning) rather
than degrading the whole array to an opaque type. When every type is present the
result is identical to a plain load. Requires `JLD2` to be loaded.

See [`toolsarray_typemap`](@ref) for the underlying typemap, which can be passed
directly to `JLD2.load`/`jldopen` (e.g. for multi-key files).
"""
function loadtoolsarray end

"""
    toolsarray_typemap(f, path, params)

A JLD2 `typemap` function that reconstructs stored `ToolsArray`/`DimArray` values
robustly (see [`loadtoolsarray`](@ref)). Pass it to either access pattern:

```julia
JLD2.load(file, key; typemap = toolsarray_typemap)
jldopen(file; typemap = toolsarray_typemap) do f; f[key]; end
```

Requires `JLD2` to be loaded.
"""
function toolsarray_typemap end

"""
    savetimeseries(file, x)

Write the time series `x` to `file`, choosing the format from its extension.

`.jld2` stores `x` exactly; it is the only lossless option. `.tsv` writes a four-line
`#` header (name, metadata, reference dimensions, variable names) followed by a table
whose first column is time. The text format makes concessions:

- Units are stripped from the cells and recorded in the metadata header, then restored
  by [`loadtimeseries`](@ref). A `Dates` time column travels the same way: instants are
  written as ISO 8601 and periods as bare counts, with the type recorded in the header.
- Reference dimensions are written but not read back.
- A series of three or more dimensions is flattened to a `DimTable` and cannot be
  reloaded; use `.jld2` for those.

## See also
- [`loadtimeseries`](@ref), [`loadtoolsarray`](@ref)
"""
savetimeseries(f::String, x) = savetimeseries(f |> query, x)

"""
    loadtimeseries(file)

Read a time series written by [`savetimeseries`](@ref), choosing the format from the
file's extension.

A `.jld2` file round trips exactly. A `.tsv` file returns the data and time values
intact, with units restored from the header, but the time lookup comes back as a vector
rather than a range, so the result is an [`IrregularTimeseries`](@ref) even when the
original was regular. Reference dimensions are dropped with a warning.

Throws an `ArgumentError` for an empty file, or for the flattened layout written for
series of three or more dimensions.

## See also
- [`savetimeseries`](@ref), [`loadtoolsarray`](@ref)
"""
loadtimeseries(f::String) = loadtimeseries(f |> query)

## JLD2 files are easiest
function savetimeseries(f::File{format"JLD2"}, x::AbstractTimeseries)
    return save(f, Dict("timeseries" => x))
end
loadtimeseries(f::File{format"JLD2"}) = load(f, "timeseries")

## Text files are harder. We can't fully reconstruct a generic timeseries, so need to make some concessions.
# We'll assume that the first column is the time index, and the remaining columns are the data.

# A TSV cell holds a bare number, so `writedlm` would otherwise emit "1.0 V" and `readdlm`
# would read it back as a string. Units instead travel as one reserved key in the metadata
# header and are re-attached on load; files without units are byte-identical to before.
const UNITSKEY = "__units__"

# A `Dates` time column travels the same way units do: recorded in the header, restored
# on load. `writedlm` writes an instant as ISO 8601, which parses back, but a `Period`
# prints as "1 day", which does not, so periods go out as bare counts.
const TIMETYPEKEY = "__timetype__"

const TIMETYPES = Dict{String, Type}(
    string(nameof(T)) => T for T in (
            Dates.DateTime, Dates.Date,
            Dates.Nanosecond, Dates.Microsecond, Dates.Millisecond, Dates.Second,
            Dates.Minute, Dates.Hour, Dates.Day, Dates.Week,
            Dates.Month, Dates.Quarter, Dates.Year,
        )
)

_tsvtimes(t) = t
_tsvtimes(t::AbstractVector{<:Dates.Period}) = Dates.value.(t)

_applytimetype(x, ::Nothing) = x
function _applytimetype(x, tt)
    T = get(TIMETYPES, tt, nothing)
    isnothing(T) && throw(
        ArgumentError(
            "loadtimeseries: unrecognised time type \"$tt\" in the file header"
        )
    )
    t = times(x)
    return set(x, 𝑡 => T <: Dates.Period ? T.(Int64.(t)) : T.(string.(t)))
end

_mdpairs(md) = md isa DimensionalData.Dimensions.NoMetadata ? Pair{Symbol, Any}[] :
    md isa DimensionalData.Metadata ? collect(pairs(md.val)) : collect(pairs(md))

_applyunits(x, ::Nothing) = x
function _applyunits(x, units)
    haskey(units, "𝑡") && (x = set(x, 𝑡 => times(x) .* uparse(units["𝑡"])))
    haskey(units, "data") && (x = x .* uparse(units["data"]))
    return x
end

function savetimeseries(f::File{format"TSV"}, x::AbstractTimeseries, var)
    isnothing(var) && (var = "")
    units = Dict{String, String}()
    _unit(eltype(x)) == NoUnits || (units["data"] = string(_unit(eltype(x))))
    _unit(eltype(times(x))) == NoUnits || (units["𝑡"] = string(_unit(eltype(times(x)))))
    md = _mdpairs(metadata(x))
    if !isempty(units)
        md = [md..., Symbol(UNITSKEY) => units]
        x = ustripall(x)
    end
    T = eltype(times(x))
    if T <: Dates.AbstractTime
        _diffunit(T) # reject a lookup type the load path cannot rebuild
        md = [md..., Symbol(TIMETYPEKEY) => string(nameof(T))]
    end
    return open(f.filename, "w") do f
        print(f, "# ")

        if name(x) isa DimensionalData.NoName
            print(f, "")
        else
            try
                print(f, json(name(x)))
            catch e
                @warn "Cannot serialize type" exception = (e, catch_backtrace())
            end
        end

        print(f, "\n# ")

        if isempty(md)
            print(f, "")
        else
            try
                print(f, json(Dict(md)))
            catch e
                @warn "Cannot serialize type" exception = (e, catch_backtrace())
            end
        end

        print(f, "\n# ")
        try
            refs = isempty(refdims(x)) ? "" : json(refdims(x))
            print(f, refs)
        catch e
            print(f, "")
            @warn "Cannot serialize type" exception = (e, catch_backtrace())
        end

        print(f, "\n# ")

        try
            vars = ndims(x) == 1 ? ["𝑡"] : json(["𝑡", name(var)])
        catch e
            vars = json(["𝑡", 1:length(var)]) # egh
            @warn "Cannot serialize type" exception = (e, catch_backtrace())
        end

        print(f, vars)
        vars = join(var, '\t')
        print(f, "\n𝑡\t$vars\n")
        writedlm(f, [_tsvtimes(times(x)) x.data], '\t')
    end
end
function savetimeseries(f::File{format"TSV"}, x::UnivariateTimeseries)
    rds = refdims(x)
    d = length(rds) == 1 ? only(rds) : refdims(x, Var)
    # Label the one data column with the refdim's *values*. Passing the dimension itself
    # puts its whole type into the header row, since `join` prints what it is given.
    v = isnothing(d) ? "" : val(d)
    var = v isa AbstractArray || v == "" ? v : [v]
    return savetimeseries(f, x, var)
end
function savetimeseries(f::File{format"TSV"}, x::MultivariateTimeseries)
    return savetimeseries(f, x, dims(x, 2))
end

function savetimeseries(f::File{format"TSV"}, x::MultidimensionalTimeseries)
    # For multidimensional time series, we have to flatten to a table to save in csv format
    return if ndims(x) < 3
        savetimeseries(f, x, dims(x, 2))
    else
        d = DimTable(x)
        save(f, d)
        # * Add metadata
        open(f.filename, "a") do f
            print(f, "# ")

            if name(x) isa DimensionalData.NoName
                print(f, "")
            else
                try
                    print(f, json(name(x)))
                catch e
                    @warn "Cannot serialize name" exception = (e, catch_backtrace())
                end
            end

            print(f, "\n# ")

            if metadata(x) isa DimensionalData.Dimensions.NoMetadata
                print(f, "")
            else
                try
                    print(f, json(metadata(x)))
                catch e
                    @warn "Cannot serialize metadata" exception = (e, catch_backtrace())
                end
            end

            print(f, "\n# ")
            try
                refs = isempty(refdims(x)) ? "" : json(refdims(x))
                print(f, refs)
            catch e
                print(f, "")
                @warn "Cannot serialize refdims" exception = (e, catch_backtrace())
            end

            print(f, "\n")
        end
    end
end

# Reading the flattened DimTable layout (written for ≥3-dimensional series) back into a
# time series is not implemented. Fail loudly rather than silently returning `nothing`.
function loadmultidimensionaltimeseries(::File{format"TSV"})
    throw(
        ArgumentError(
            "loadtimeseries: reading a multidimensional (≥3D) time series " *
                "from TSV is not supported; save it as JLD2 instead."
        )
    )
end

function loadtimeseries(f::File{format"TSV"})
    line = readline(f.filename)
    isempty(line) &&
        throw(ArgumentError("loadtimeseries: $(f.filename) is empty"))
    if first(line) != '#'
        return loadmultidimensionaltimeseries(f)
    end
    return open(f.filename, "r") do f
        # Read the name
        line = readline(f)
        name = isempty(line[3:end]) ? DimensionalData.NoName() : JSON.parse(line[3:end])

        # Read the metadata
        line = readline(f)
        metadata = isempty(line[3:end]) ?
            DimensionalData.Dimensions.NoMetadata() :
            JSON.parse(line[3:end])
        units = nothing
        timetype = nothing
        if metadata isa AbstractDict
            metadata = Dict{String, Any}(metadata) # JSON.Object is immutable
            units = pop!(metadata, UNITSKEY, nothing)
            timetype = pop!(metadata, TIMETYPEKEY, nothing)
            isempty(metadata) && (metadata = DimensionalData.Dimensions.NoMetadata())
        end

        # Read the reference dimensions
        line = readline(f)
        if !isempty(line[3:end])
            @warn "Cannot load refdims yet"
        end
        refdims = () # Not implemented, maybe isempty(line[3:end]) ? () : JSON.parse(line[3:end])

        # Read the variable names
        line = readline(f)
        j = JSON.parse(line[3:end])
        vars = length(j) > 1 ? j[2] : ()
        if vars == "Var"
            vars = Var
        elseif vars == "X"
            vars = X
        elseif vars == "Y"
            vars = Y
        elseif vars == "Z"
            vars = Z
        elseif vars == "𝑓"
            vars = 𝑓
        end

        # Read the variables
        line = readline(f)
        j = Meta.parse.(split(line, '\t')[2:end])
        if vars isa Type
            vars = vars(j)
        elseif !isempty(vars)
            vars = Dim{Symbol(vars)}(j)
        end

        # `readdlm` with `header=false` always returns a matrix; assert it so the type
        # is concrete (and so callers below don't see the header-tuple union it declares).
        data = readdlm(f, '\t', header = false)::Matrix
        # An instant column makes `readdlm` return a `Matrix{Any}`, which would leave
        # the *data* as `Vector{Any}` too; `identity.` narrows it back.
        tcol = data[:, 1]
        vals = identity.(data[:, 2:end])
        if isempty(vars)
            x = Timeseries(vals[:, 1], tcol; name, metadata, refdims)
        else
            x = Timeseries(vals, 𝑡(tcol), vars; name, metadata, refdims)
        end
        return _applyunits(_applytimetype(x, timetype), units)
    end
end

"""
    savets

Alias for [`savetimeseries`](@ref).
"""
savets = savetimeseries

"""
    loadts

Alias for [`loadtimeseries`](@ref).
"""
loadts = loadtimeseries

end
