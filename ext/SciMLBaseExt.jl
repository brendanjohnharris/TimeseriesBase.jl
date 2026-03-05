module SciMLBaseExt
using SciMLBase
import SciMLBase: AbstractTimeseriesSolution, AbstractEnsembleSolution
using TimeseriesBase
import TimeseriesBase: Timeseries

function sol2metadata(sol; kwargs...) # Maybe expand
    Dict(:alg => sol.alg,
         :alg_choice => sol.alg_choice,
         :retcode => sol.retcode,
         pairs(kwargs)...)
end

function split_duplicates(t::AbstractVector)
    last_idx = Dict{eltype(t), Int}()
    dup_idx = Int[]

    for (i, time) in enumerate(t)
        haskey(last_idx, time) && push!(dup_idx, last_idx[time])
        last_idx[time] = i
    end

    return sort!(collect(values(last_idx))), dup_idx
end

function split_duplicate_timesteps(u, t)
    unique_idx, dup_idx = split_duplicates(t)

    return u[unique_idx], t[unique_idx], u[dup_idx], t[dup_idx]
end

function split_duplicate_timesteps(u, t::AbstractRange)
    return u, t, similar(u, 0), similar(t, 0)
end

function Timeseries(sol::AbstractTimeseriesSolution{T, 1, V}) where {T, V}
    u, t, du, dt = split_duplicate_timesteps(sol.u, sol.t)
    callback_values = Timeseries(du, dt)
    Timeseries(u, t;
               metadata = sol2metadata(sol; callback_values))::AbstractTimeseries{T, 1}
end

function Timeseries(sol::AbstractTimeseriesSolution{T, 2, V}) where {T, V}
    vars = 1:length(first(sol.u))
    u, t, du, dt = split_duplicate_timesteps(sol.u, sol.t)
    if !isempty(du)
        callback_values = Timeseries(permutedims(stack(du), (2, 1)), dt, vars)
    else
        callback_values = Timeseries([], [])
    end
    Timeseries(permutedims(stack(u), (2, 1)), t, vars;
               metadata = sol2metadata(sol; callback_values))::AbstractTimeseries{T, 2}
end

function Timeseries(sol::AbstractTimeseriesSolution{T, N, <:AbstractVector{V}}) where {T, N,
                                                                                       V
                                                                                       }
    u, t, du, dt = split_duplicate_timesteps(sol.u, sol.t)
    callback_values = Timeseries(du, dt)
    Timeseries(u, t;
               metadata = sol2metadata(sol; callback_values))::AbstractTimeseries{V, 1}
end

function Timeseries(sol::AbstractEnsembleSolution{T, N, S};
                    dims = Obs(1:length(sol))) where {T, N, S}
    ToolsArray(map(Timeseries, sol), dims)
end

end # module
