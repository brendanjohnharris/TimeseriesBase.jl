module DatesTools

using TimeseriesBase.ToolsArrays
using DimensionalData

import DimensionalData.Dates

export DateIndex, DateTimeIndex, DateTimeseries

DateIndex = DateTIndex = Union{
    AbstractArray{<:Dates.AbstractTime},
    AbstractRange{<:Dates.AbstractTime},
    Tuple{<:Dates.AbstractTime},
}

DateTimeIndex = Tuple{
    A,
    Vararg{DimensionalData.Dimension},
} where {
    A <:
    DimensionalData.Dimension{<:DateIndex},
}

DateTimeseries = AbstractToolsArray{T, N, <:DateTimeIndex, B} where {T, N, B}

end
