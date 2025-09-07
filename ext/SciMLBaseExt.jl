module SciMLBaseExt
using SciMLBase
import SciMLBase: AbstractTimeseriesSolution, AbstractEnsembleSolution
using TimeseriesBase
import TimeseriesBase: Timeseries

function sol2metadata(sol) # Maybe expand
    Dict(:alg => sol.alg,
         :alg_choice => sol.alg_choice,
         :retcode => sol.retcode)
end

function Timeseries(sol::AbstractTimeseriesSolution{T, 1, V}) where {T, V}
    Timeseries(sol.u, sol.t; metadata = sol2metadata(sol))::AbstractTimeseries{T, 1}
end

function Timeseries(sol::AbstractTimeseriesSolution{T, 2, V}) where {T, V}
    Timeseries(permutedims(stack(sol.u), (2, 1)), sol.t, 1:length(first(sol.u));
               metadata = sol2metadata(sol))::AbstractTimeseries{T, 2}
end

function Timeseries(sol::AbstractTimeseriesSolution{T, N, <:AbstractVector{V}}) where {T, N,
                                                                                       V
                                                                                       }
    Timeseries(sol.u, sol.t; metadata = sol2metadata(sol))::AbstractTimeseries{V, 1}
end

function Timeseries(sol::AbstractEnsembleSolution{T, N, S};
                    dims = Obs(1:length(sol))) where {T, N, S}
    ToolsArray(map(Timeseries, sol), dims)
end

end # module
