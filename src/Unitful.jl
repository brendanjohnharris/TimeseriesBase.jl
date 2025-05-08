using Unitful
import Unitful.unit

export dimunit, timeunit, frequnit, unit,
       UnitfulIndex, UnitfulTimeSeries, UnitfulSpectrum,
       ustripall

# Unitful._promote_unit(::S, ::T) where {S<:Unitful.FreeUnits{(), NoDims, nothing}, T<:Unitful.TimeUnits} = u"s"
"""
    TimeseriesToolsBase.convertconst(c::Number, u::Unitful.Quantity)

Converts a constant `c` to have the same units as `u`.

## Examples
```@example 1
julia> using Unitful;
julia> c = 5;
julia> u = 3u"s";
julia> converted_c = TimeseriesToolsBase.convertconst(c, u);
julia> typeof(converted_c) == typeof(u)
```
"""
TimeseriesToolsBase.convertconst(c::Number, u::Unitful.Quantity) = (c)unit(u)

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
    UnitfulTimeSeries{T, N, B}

A type alias for an `AbstractToolsArray` with a [`UnitfulTimeIndex`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = (1:100)u"s";
julia> x = rand(100);
julia> uts = TimeSeries(t, x);
julia> uts isa UnitfulTimeSeries
```
"""
UnitfulTimeSeries = AbstractToolsArray{T, N, <:UnitfulTimeIndex, B} where {T, N, B}

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

function unitfultimeseries(x::AbstractTimeSeries, u::Unitful.Units)
    t = x |> times
    t = timeunit(x) == NoUnits ? t : ustrip(t)
    t = t * u
    ds = dims(x)
    return ToolsArray(x.data, (𝑡(t), ds[2:end]...); metadata = DimensionalData.metadata(x),
                      name = DimensionalData.name(x), refdims = DimensionalData.refdims(x))
end

function unitfultimeseries(x::AbstractTimeSeries)
    if timeunit(x) == NoUnits
        @warn "No time units found for unitful time series. Assuming seconds."
        return unitfultimeseries(x, u"s")
    else
        return x
    end
end

"""
    TimeSeries(t, x, timeunit::Unitful.Units)

Constructs a univariate time series with time `t`, data `x` and time units specified by `timeunit`.
Note that you can add units to the elements of a time series `x` with, for example, `x*u"V"`.

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = TimeSeries(t, x, u"ms")*u"V";
julia> ts isa Union{UnivariateTimeSeries, RegularTimeSeries, UnitfulTimeSeries}
```
"""
TimeSeries(t, x, unit::Unitful.Units) = TimeSeries((t)unit, x)
TimeSeries(t, v, x, unit::Unitful.Units) = TimeSeries((t)unit, v, x)

"""
    dimunit(x::UnitfulTimeSeries, dim)

Returns the unit associated with the specified dimension `dim` of a [`UnitfulTimeSeries`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = TimeSeries(t, x, u"ms");
julia> TimeseriesToolsBase.dimunit(ts, 𝑡) == u"ms"
```
"""
dimunit(x::AbstractToolsArray, dim) = dims(x, dim) |> eltype |> unit

"""
    timeunit(x::UnitfulTimeSeries)

Returns the time units associated with a [`UnitfulTimeSeries`].

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = TimeSeries(t, x, u"ms");
julia> timeunit(ts) == u"ms"
```
"""
timeunit(x::AbstractTimeSeries) = dimunit(x, 𝑡)

"""
    frequnit(x::UnitfulSpectrum)

Returns the frequency units associated with a [`UnitfulSpectrum`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = TimeSeries(t, x, u"ms");
julia> sp = fft(ts);  # assuming fft returns a UnitfulSpectrum
julia> frequnits(sp) == u"Hz"
```
"""
frequnit(x::AbstractSpectrum) = dimunit(x, 𝑓)

"""
    unit(x::AbstractArray)

Returns the units associated with the elements of an [`UnitfulTimeSeries`](@ref) or [`UnitfulSpectrum`](@ref).

## Examples
```@example 1
julia> using Unitful;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = TimeSeries(t, x, u"ms")*u"V";
julia> unit(ts) == u"V"
```
"""
unit(x::Union{<:AbstractTimeSeries, AbstractSpectrum}) = x |> eltype |> unit
unit(x::Union{<:AbstractTimeSeries{Any}, AbstractSpectrum{Any}}) = NoUnits


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
ustripall(x::AbstractArray{T}) where {T <: Number} = ustrip.(x)
ustripall(a::AbstractRange) = a
ustripall(a::AbstractRange{<:Quantity}) = ustrip.(a)
ustripall(a::ClosedInterval) = ustrip(a.left) .. ustrip(a.right)
ustripall(x::AbstractVector{<:AbstractString}) = x
ustripall(x::AbstractVector{<:Symbol}) = x
ustripall(x) = ustrip(x)
