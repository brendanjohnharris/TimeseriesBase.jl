module IO

import FileIO: save, load, @format_str, File, query
using JSON
import JSON.json
using DelimitedFiles
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.Utils
using DimensionalData

export savetimeseries, savets, loadtimeseries, loadts

savetimeseries(f::String, x) = savetimeseries(f |> query, x)
loadtimeseries(f::String) = loadtimeseries(f |> query)

## JLD2 files are easiest
function savetimeseries(f::File{format"JLD2"}, x::AbstractTimeseries)
    save(f, Dict("timeseries" => x))
end
loadtimeseries(f::File{format"JLD2"}) = load(f, "timeseries")

## Text files are harder. We can't fully reconstruct a generic timeseries, so need to make some concessions.
# We'll assume that the first column is the time index, and the remaining columns are the data.
function savetimeseries(f::File{format"TSV"}, x::AbstractTimeseries, var)
    isnothing(var) && (var = "")
    open(f.filename, "w") do f
        print(f, "# ")

        if name(x) isa DimensionalData.NoName
            print(f, "")
        else
            try
                print(f, json(name(x)))
            catch e
                @warn "Cannot serialize type" exception=(e, catch_backtrace())
            end
        end

        print(f, "\n# ")

        if metadata(x) isa DimensionalData.Dimensions.NoMetadata
            print(f, "")
        else
            try
                print(f, json(metadata(x)))
            catch e
                @warn "Cannot serialize type" exception=(e, catch_backtrace())
            end
        end

        print(f, "\n# ")
        try
            refs = isempty(refdims(x)) ? "" : json(refdims(x))
            print(f, refs)
        catch e
            print(f, "")
            @warn "Cannot serialize type" exception=(e, catch_backtrace())
        end

        print(f, "\n# ")

        try
            vars = ndims(x) == 1 ? ["𝑡"] : json(["𝑡", name(var)])
        catch e
            vars = json(["𝑡", 1:length(var)]) # egh
            @warn "Cannot serialize type" exception=(e, catch_backtrace())
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
    savetimeseries(f, x, var)
end
function savetimeseries(f::File{format"TSV"}, x::MultivariateTimeseries)
    savetimeseries(f, x, dims(x, 2))
end

function savetimeseries(f::File{format"TSV"}, x::MultidimensionalTimeseries)
    # For multidimensional time series, we have to flatten to a table to save in csv format
    if ndims(x) < 3
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
                    @warn e
                end
            end

            print(f, "\n# ")

            if metadata(x) isa DimensionalData.Dimensions.NoMetadata
                print(f, "")
            else
                try
                    print(f, json(metadata(x)))
                catch e
                    @warn e
                end
            end

            print(f, "\n# ")
            try
                refs = isempty(refdims(x)) ? "" : json(refdims(x))
                print(f, refs)
            catch e
                print(f, "")
                @warn e
            end

            print(f, "\n")
        end
    end
end
function loadmultidimensionaltimeseries(f::File{format"TSV"})
    d = load(f)
    # table2timeseries(D) # ! Todo...
end

function loadtimeseries(f::File{format"TSV"})
    if first(readline(f.filename)) != '#'
        return loadmultidimensionaltimeseries(f)
    end
    open(f.filename, "r") do f
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

        data = readdlm(f, '\t', header = false)
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
