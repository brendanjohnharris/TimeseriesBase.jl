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

"""
    DateTimeseries{T, N, B}

A type alias for an `AbstractToolsArray` whose time lookup holds `Dates` values.

!!! note
    [`DateIndex`](@ref) is built on `Dates.AbstractTime`, which covers both *instants*
    (`Dates.TimeType`: `DateTime`, `Date`) and *durations* (`Dates.Period`: `Day`,
    `Millisecond`, ...). Both satisfy this alias, but they behave differently: an
    instant axis is a calendar, so it supports `groupby(𝑡 => months(1))` and instant
    selectors, while a duration axis is elapsed time from an implicit origin.
    [`stitch`](@ref) preserves an instant axis, and `regularize(...; zero = true)`
    deliberately returns a duration axis.

    `Dates.Time` is not supported: it is a time of day, it wraps at midnight, and its
    ranges are unreliable in `Base`.
"""
DateTimeseries = AbstractToolsArray{T, N, <:DateTimeIndex, B} where {T, N, B}

"""
    TimeseriesBase.DatesTools._diffunit(::Type{<:Dates.AbstractTime})

The `Period` type in which differences of a lookup are exactly representable: the
resolution of an instant type, or a duration type itself. Any other `Dates` type is
rejected, since the package has no lossless integer representation for it.
"""
_diffunit(::Type{Dates.DateTime}) = Dates.Millisecond
_diffunit(::Type{Dates.Date}) = Dates.Day
function _diffunit(::Type{P}) where {P <: Dates.Period}
    # `P <: Dates.Period` matches the abstract type too, which a mixed-unit lookup such
    # as `Period[Day(1), Hour(2)]` has: its differences are `CompoundPeriod`s, which
    # carry no single count.
    isconcretetype(P) || throw(
        ArgumentError(
            "a lookup of mixed Period types has no single resolution to count in; " *
                "convert it to one period type first"
        )
    )
    return P
end
function _diffunit(::Type{T}) where {T <: Dates.AbstractTime}
    return throw(
        ArgumentError(
            "a $T lookup is not supported; use DateTime, Date, or a fixed Period"
        )
    )
end

"""
    TimeseriesBase.DatesTools._offsets(lk, t0 = first(lk))

The lookup `lk` as `Int64` counts of [`_diffunit`](@ref)`(eltype(lk))`, measured from
`t0`. Exact: the conversion is always to a finer unit, so the round trip through
[`_reinstate`](@ref) is lossless. A range in gives a range out, so regularity survives.
"""
function _offsets(lk::AbstractRange{<:Dates.AbstractTime}, t0 = first(lk))
    P = _diffunit(eltype(lk))
    return range(
        Dates.value(convert(P, first(lk) - t0)),
        step = Dates.value(convert(P, Base.step(lk))),
        length = length(lk)
    )
end
function _offsets(lk, t0 = first(lk))
    P = _diffunit(eltype(lk))
    return Int64[Dates.value(convert(P, t - t0)) for t in lk]
end

"""
    TimeseriesBase.DatesTools._reinstate(t0, v)

Invert [`_offsets`](@ref): `v` counts of [`_diffunit`](@ref)`(typeof(t0))` measured from
`t0`, rounded to the nearest whole count (`RoundNearest`, so ties go to even). A range in
gives a range out.
"""
_reinstate(t0, v::Number) = t0 + _diffunit(typeof(t0))(round(Int64, v))
function _reinstate(t0, v::AbstractRange)
    P = _diffunit(typeof(t0))
    return range(
        _reinstate(t0, first(v)),
        step = P(round(Int64, Base.step(v))),
        length = length(v)
    )
end
function _reinstate(t0, v::AbstractVector)
    P = _diffunit(typeof(t0))
    return [t0 + P(round(Int64, x)) for x in v]
end

"""
    TimeseriesBase.DatesTools._onoffsets(f, lk)

Apply `f` to the values of `lk` expressed as numeric offsets from `first(lk)`, then map
the result back. Instants do not add and periods do not divide exactly, so every average,
midpoint and interpolation of a `Dates` lookup goes through here. A lookup that is not a
`Dates` lookup is handed to `f` untouched.
"""
_onoffsets(f, lk) = f(lk)
function _onoffsets(f, lk::AbstractVector{<:Dates.AbstractTime})
    return _reinstate(first(lk), f(_offsets(lk)))
end

end
