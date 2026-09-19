module ToolsArrays

using DimensionalData
import DimensionalData: Dimension, NoName, NoMetadata, format, AbstractBasicDimArray

export AbstractToolsArray, ToolsArray,
    ToolsDimension, ToolsDim,
    𝑡, 𝑥, 𝑦, 𝑧, 𝑓, Var, Obs, Log𝑓, Log10𝑓

"""
A local type to avoid overloading and piracy issues with DimensionalData.jl
"""
abstract type AbstractToolsArray{T, N, D, A} <: DimensionalData.AbstractDimArray{T, N, D, A} end

AbstractDimVector = AbstractToolsArray{T, 1} where {T}
AbstractDimMatrix = AbstractToolsArray{T, 2} where {T}

# struct ToolsArray{T, N, D <: Tuple, R <: Tuple, A <: AbstractArray{T, N}, Na, Me} <:
#        AbstractToolsArray{T, N, D, A}
#     data::A
#     dims::D
#     refdims::R
#     name::Na
#     metadata::Me
# end

## ? Constructors: see DimensionalData.jl/array/array.jl
struct ToolsArray{T, N, D <: Tuple, R <: Tuple, A <: AbstractArray{T, N}, Na, Me} <:
    AbstractToolsArray{T, N, D, A}
    data::A
    dims::D
    refdims::R
    name::Na
    metadata::Me

    function ToolsArray(
            data::A, dims::D, refdims::R, name::Na,
            metadata::Me
        ) where {
            D <: Tuple, R <: Tuple,
            A <: AbstractArray{T, N},
            Na, Me,
        } where {T, N}
        DimensionalData.checkdims(data, dims)
        return new{T, N, D, R, A, Na, Me}(data, dims, refdims, name, metadata)
    end

    # * If the parent array is a AbstractDimArray, recurse until we hit the root array
    function ToolsArray(
            data::A, dims::D, refdims::R, name::Na,
            metadata::Me
        ) where {
            D <: Tuple, R <: Tuple,
            A <: AbstractDimArray{T, N, d, a},
            Na, Me,
        } where {T, N, d, a}
        DimensionalData.checkdims(parent(data), dims)
        return new{T, N, D, R, a, Na, Me}(parent(data), dims, refdims, name, metadata)
    end
end
# 2 arg version
ToolsArray(data::AbstractArray, dims; kw...) = ToolsArray(data, (dims,); kw...)
function ToolsArray(
        data::AbstractArray, dims::Union{Tuple, NamedTuple};
        refdims = (), name = NoName(), metadata = NoMetadata()
    )
    return ToolsArray(data, format(dims, data), refdims, name, metadata)
end
function ToolsArray(data::AbstractArray, dims::Vararg{Dimension}; kwargs...)
    return ToolsArray(data, dims; kwargs...)
end
# All keyword argument version
function ToolsArray(; data, dims, refdims = (), name = NoName(), metadata = NoMetadata())
    return ToolsArray(data, dims; refdims, name, metadata)
end
# Construct from another AbstractDimArray
function ToolsArray(
        A::AbstractDimArray;
        data = parent(A), dims = dims(A), refdims = refdims(A), name = name(A),
        metadata = metadata(A)
    )
    return ToolsArray(data, dims; refdims, name, metadata)
end
function ToolsArray(
        A::AbstractBasicDimArray; data = parent(A), dims = dims(A),
        kwargs...
    )
    return ToolsArray(data, dims; kwargs...)
end
# Convert only the element type: broadcasting `convert` would drop name and metadata,
# and the keyword arguments must still reach the constructor.
function ToolsArray{T}(A::AbstractToolsArray; kw...) where {T}
    return ToolsArray(A; data = convert.(T, parent(A)), kw...)
end
ToolsArray{T}(A::AbstractToolsArray{T}; kw...) where {T} = ToolsArray(A; kw...)

"""
    ToolsArray(f::Function, dim::Dimension; [name])

Apply function `f` across the values of the dimension `dim`
(using `broadcast`), and return the result as a dimensional array with
the given dimension. Optionally provide a name for the result.
"""
function ToolsArray(
        f::Function, dim::Dimension;
        name = Symbol(nameof(f), "(", name(dim), ")"), kwargs...
    )
    return ToolsArray(map(f, val(dim)), (dim,); name, kwargs...)
end
function ToolsArray(
        f::Function, dims::Vararg{Dimension};
        name = Symbol(nameof(f), "(", join(name.(dims), ','), ")"),
        kwargs...
    )
    data = map(Iterators.product(map(val, dims)...)) do args
        f(args...)
    end
    return ToolsArray(data, dims; name, kwargs...)
end

## Extra constructors

ToolsArray(x::AbstractArray, D::DimensionalData.Dimension) = ToolsArray(x, (D,))
function ToolsArray(D::DimensionalData.DimArray)
    return ToolsArray(D.data, D.dims, D.refdims, D.name, D.metadata)
end

@inline function DimensionalData.rebuild(
        A::ToolsArray, data::AbstractArray, dims::Tuple,
        refdims::Tuple, name, metadata
    )
    return ToolsArray(data, dims, refdims, name, metadata)
end

# * Custom dimensions
import DimensionalData: TimeDim, XDim, YDim, ZDim
DimensionalData.@dim 𝑡 TimeDim "Time"
DimensionalData.@dim 𝑥 XDim "x"
DimensionalData.@dim 𝑦 YDim "y"
DimensionalData.@dim 𝑧 ZDim "z"

abstract type VariableDim{T} <: Dimension{T} end
DimensionalData.@dim Var VariableDim "Var"

abstract type ObservationDim{T} <: Dimension{T} end
DimensionalData.@dim Obs ObservationDim "Obs"

abstract type FrequencyDim{T} <: Dimension{T} end
DimensionalData.@dim 𝑓 FrequencyDim "Frequency"

abstract type LogFrequencyDim{T} <: Dimension{T} end
DimensionalData.@dim Log10𝑓 LogFrequencyDim "Log10 Frequency"
DimensionalData.@dim Log𝑓 LogFrequencyDim "Natural Log Frequency"

"""
    𝑡

A DimensionalData.jl dimension representing time. An array whose first dimension is `𝑡`
is an [`AbstractTimeseries`](@ref); its values are returned by [`times`](@ref).

## See also
- [`Timeseries`](@ref), [`times`](@ref), [`samplingrate`](@ref)
"""
𝑡

"""
    𝑥

A DimensionalData.jl dimension representing the first spatial axis.

## See also
- [`𝑦`](@ref), [`𝑧`](@ref)
"""
𝑥

"""
    𝑦

A DimensionalData.jl dimension representing the second spatial axis.

## See also
- [`𝑥`](@ref), [`𝑧`](@ref)
"""
𝑦

"""
    𝑧

A DimensionalData.jl dimension representing the third spatial axis.

## See also
- [`𝑥`](@ref), [`𝑦`](@ref)
"""
𝑧

"""
    Obs

A DimensionalData.jl dimension representing repeated observations of the same quantity
(trials, repeats, or realisations), as distinct from [`Var`](@ref), which indexes
different quantities.
"""
Obs

"""
    Log10𝑓

A DimensionalData.jl dimension representing frequency on a base-10 logarithmic scale; its
values are `log10(f)`, not `f`.

!!! note
    `Log10𝑓` is a `LogFrequencyDim`, not a `FrequencyDim`, so an array indexed by it is
    not an `AbstractSpectrum` and [`freqs`](@ref) does not apply to it.

## See also
- [`𝑓`](@ref), [`Log𝑓`](@ref)
"""
Log10𝑓

"""
    Log𝑓

A DimensionalData.jl dimension representing frequency on a natural logarithmic scale; its
values are `log(f)`, not `f`.

!!! note
    `Log𝑓` is a `LogFrequencyDim`, not a `FrequencyDim`, so an array indexed by it is not
    an `AbstractSpectrum` and [`freqs`](@ref) does not apply to it.

## See also
- [`𝑓`](@ref), [`Log10𝑓`](@ref)
"""
Log𝑓

"""
    ToolsDim{T}
An abstract type for custom macro-defined dimensions in `TimeseriesBase`. Analogous to
`DimensionalData.Dimension` for the purposes of `DimensionalData.@dim`.

## Examples
```
DimensionalData.@dim MyDim ToolsDim "My dimension" # Defines a new `ToolsDim <: ToolsDimension`
```

## See also
- [`ToolsDimension`](@ref)
"""
abstract type ToolsDim{T} <: DimensionalData.Dimension{T} end

"""
    ToolsDimension
A union of all `Dimension` types that fall within the scope of `TimeseriesBase`. Analogous
to `DimensionalData.Dimension` for dispatch purposes.

## See also
- [`ToolsDim`](@ref)
"""
ToolsDimension = Union{𝑡, 𝑥, 𝑦, 𝑧, 𝑓, Log𝑓, Log10𝑓, Var, Obs, ToolsDim}

function DimensionalData.dimconstructor(
        ::Tuple{
            ToolsDimension,
            Vararg{DimensionalData.Dimension},
        }
    )
    return ToolsArray
end
DimensionalData.dimconstructor(::Tuple{<:ToolsDimension, Vararg}) = ToolsArray
DimensionalData.dimconstructor(dims::ToolsDimension) = ToolsArray

end
