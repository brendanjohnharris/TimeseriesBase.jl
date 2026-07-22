module IO

import FileIO: save, load, @format_str, File, query
using JSON
import JSON.json
using DelimitedFiles
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.Utils
using DimensionalData

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

savetimeseries(f::String, x) = savetimeseries(f |> query, x)
loadtimeseries(f::String) = loadtimeseries(f |> query)

## JLD2 files are easiest
function savetimeseries(f::File{format"JLD2"}, x::AbstractTimeseries)
    return save(f, Dict("timeseries" => x))
end
loadtimeseries(f::File{format"JLD2"}) = load(f, "timeseries")

## Text files are harder. We can't fully reconstruct a generic timeseries, so need to make some concessions.
# We'll assume that the first column is the time index, and the remaining columns are the data.
function savetimeseries(f::File{format"TSV"}, x::AbstractTimeseries, var)
    isnothing(var) && (var = "")
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

        if metadata(x) isa DimensionalData.Dimensions.NoMetadata
            print(f, "")
        else
            try
                print(f, json(metadata(x)))
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
        writedlm(f, [times(x) x.data], '\t')
    end
end
function savetimeseries(f::File{format"TSV"}, x::UnivariateTimeseries)
    if length(refdims(x)) == 1
        var = refdims(x)
    else
        var = refdims(x, Var)
    end
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
    if first(readline(f.filename)) != '#'
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
        if isempty(vars)
            x = Timeseries(data[:, 2], data[:, 1]; name, metadata, refdims)
        else
            x = Timeseries(data[:, 2:end], 𝑡(data[:, 1]), vars; name, metadata, refdims)
        end
    end
end

savets = savetimeseries
loadts = loadtimeseries

end
