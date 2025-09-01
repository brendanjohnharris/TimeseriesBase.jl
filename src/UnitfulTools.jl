module UnitfulTools

using TimeseriesBase.TimeSeries
import TimeseriesBase.TimeSeries: convertconst, Timeseries
using TimeseriesBase.ToolsArrays
using TimeseriesBase.Spectra

using IntervalSets
using Unitful
import Unitful.unit

using DimensionalData

export dimunit, timeunit, frequnit, unit,
       UnitfulIndex, UnitfulTimeseries, UnitfulSpectrum,
       ustripall

# Unitful._promote_unit(::S, ::T) where {S<:Unitful.FreeUnits{(), NoDims, nothing}, T<:Unitful.TimeUnits} = u"s"
"""
    TimeseriesBase.Uniful.convertconst(c::Number, u::Unitful.Quantity)

Converts a constant `c` to have the same units as `u`.

## Examples
```@example 1
julia> using Unitful;
julia> c = 5;
julia> u = 3u"s";
julia> converted_c = TimeseriesBase.Unitful.convertconst(c, u);
julia> typeof(converted_c) == typeof(u)
```
"""
convertconst(c::Number, u::Unitful.Quantity) = (c)unit(u)

"""
    UnitfulIndex

A type alias for a union of `AbstractArray`, `AbstractRange`, and `Tuple` types with `Unitful.Time` elements.
"""
UnitfulIndex = UnitfulTIndex = Union{AbstractArray{<:Unitful.Time},
                                     AbstractRange{<:Unitful.Time}, Tuple{<:Unitful.Time}}

"""
    UnitfulTimeIndex

A type alias for a tuple of dimensions, where the first dimension is of type `DimensionalData.Dimension{<:UnitfulIndex}`.
"""
UnitfulTimeIndex = Tuple{A,
                         Vararg{DimensionalData.Dimension}} where {A <:
                                                                   DimensionalData.Dimension{<:UnitfulIndex}}

"""
    UnitfulTimeseries{T, N, B}

A type alias for an `AbstractToolsArray` with a [`UnitfulTimeIndex`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = (1:100)u"s";
julia> x = rand(100);
julia> uts = Timeseries(x, t);
julia> uts isa UnitfulTimeseries
```
"""
UnitfulTimeseries = AbstractToolsArray{T, N, <:UnitfulTimeIndex, B} where {T, N, B}

UnitfulFIndex = Union{AbstractArray{<:Unitful.Frequency},
                      AbstractRange{<:Unitful.Frequency}, Tuple{<:Unitful.Frequency}}
UnitfulFreqIndex = Tuple{A,
                         Vararg{DimensionalData.Dimension}} where {A <:
                                                                   DimensionalData.Dimension{<:UnitfulFIndex}}

"""
    UnitfulSpectrum{T,N,B}

A type representing spectra with unitful frequency units.
"""
UnitfulSpectrum = AbstractToolsArray{T, N, <:UnitfulFreqIndex, B} where {T, N, B}

function unitfultimeseries(x::AbstractTimeseries, u::Unitful.Units)
    t = x |> times
    t = timeunit(x) == NoUnits ? t : ustrip(t)
    t = t * u
    ds = dims(x)
    return ToolsArray(x.data, (𝑡(t), ds[2:end]...); metadata = DimensionalData.metadata(x),
                      name = DimensionalData.name(x), refdims = DimensionalData.refdims(x))
end

function unitfultimeseries(x::AbstractTimeseries)
    if timeunit(x) == NoUnits
        @warn "No time units found for unitful time series. Assuming seconds."
        return unitfultimeseries(x, u"s")
    else
        return x
    end
end

"""
    dimunit(x::UnitfulTimeseries, dim)

Returns the unit associated with the specified dimension `dim` of a [`UnitfulTimeseries`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, (t)u"ms");
julia> TimeseriesBase.dimunit(ts, 𝑡) == u"ms"
```
"""
dimunit(x::AbstractToolsArray, dim) = dims(x, dim) |> eltype |> unit

"""
    timeunit(x::UnitfulTimeseries)

Returns the time units associated with a [`UnitfulTimeseries`].

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, (t)u"ms");
julia> timeunit(ts) == u"ms"
```
"""
timeunit(x::AbstractTimeseries) = dimunit(x, 𝑡)

"""
    frequnit(x::UnitfulSpectrum)

Returns the frequency units associated with a [`UnitfulSpectrum`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, (t)u"ms");
julia> sp = fft(ts);  # assuming fft returns a UnitfulSpectrum
julia> frequnits(sp) == u"Hz"
```
"""
frequnit(x::AbstractSpectrum) = dimunit(x, 𝑓)

"""
    unit(x::AbstractArray)

Returns the units associated with the elements of an [`UnitfulTimeseries`](@ref) or [`UnitfulSpectrum`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, (t)u"ms")*u"V";
julia> unit(ts) == u"V"
```
"""
unit(x::Union{<:AbstractTimeseries, AbstractSpectrum}) = x |> eltype |> unit
unit(x::Union{<:AbstractTimeseries{Any}, AbstractSpectrum{Any}}) = NoUnits

function ustripall(x::AbstractDimArray)
    x = set(x, ustripall.(parent(x)))
    for d in dims(x)
        x = set(x, d => ustripall(lookup(x, d)))
    end
    return x
end
ustripall(d::DimensionalData.Dimension) = ustripall(parent(lookup(d)))
ustripall(d::DimensionalData.LookupArray) = ustripall(parent(d))
ustripall(x::String) = x
ustripall(x::AbstractArray{T}) where {T <: Number} = x
ustripall(x::AbstractArray{T}) where {T <: Quantity} = ustrip(x)
ustripall(a::AbstractRange) = a
ustripall(a::AbstractRange{<:Quantity}) = ustrip(a)
ustripall(a::IntervalSets.TypedEndpointsInterval) = ustrip(a.left) .. ustrip(a.right)
ustripall(x::AbstractArray{<:AbstractString}) = x
ustripall(x::AbstractArray{<:Symbol}) = x
ustripall(x) = ustrip(x)

end
