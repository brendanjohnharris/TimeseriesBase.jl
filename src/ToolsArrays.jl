module ToolsArrays

using DimensionalData
import DimensionalData: Dimension, NoName, NoMetadata, format

export AbstractToolsArray, ToolsArray,
       ToolsDimension, ToolsDim,
       𝑡, 𝑥, 𝑦, 𝑧, 𝑓, Var, Obs

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

    function ToolsArray(data::A, dims::D, refdims::R, name::Na,
                        metadata::Me) where {D <: Tuple, R <: Tuple,
                                             A <: AbstractArray{T, N},
                                             Na, Me} where {T, N}
        DimensionalData.checkdims(data, dims)
        new{T, N, D, R, A, Na, Me}(data, dims, refdims, name, metadata)
    end

    # * If the parent array is a AbstractDimArray, recurse until we hit the root array
    function ToolsArray(data::A, dims::D, refdims::R, name::Na,
                        metadata::Me) where {D <: Tuple, R <: Tuple,
                                             A <: AbstractDimArray{T, N, d, a},
                                             Na, Me} where {T, N, d, a}
        DimensionalData.checkdims(parent(data), dims)
        new{T, N, D, R, a, Na, Me}(parent(data), dims, refdims, name, metadata)
    end
end
# 2 arg version
ToolsArray(data::AbstractArray, dims; kw...) = ToolsArray(data, (dims,); kw...)
function ToolsArray(data::AbstractArray, dims::Union{Tuple, NamedTuple};
                    refdims = (), name = NoName(), metadata = NoMetadata())
    ToolsArray(data, format(dims, data), refdims, name, metadata)
end
function ToolsArray(data::AbstractArray, dims::Vararg{Dimension}; kwargs...)
    ToolsArray(data, dims; kwargs...)
end
# All keyword argument version
function ToolsArray(; data, dims, refdims = (), name = NoName(), metadata = NoMetadata())
    ToolsArray(data, dims; refdims, name, metadata)
end
# Construct from another AbstractDimArray
function ToolsArray(A::AbstractDimArray;
                    data = parent(A), dims = dims(A), refdims = refdims(A), name = name(A),
                    metadata = metadata(A))
    ToolsArray(data, dims; refdims, name, metadata)
end
ToolsArray{T}(A::AbstractToolsArray; kw...) where {T} = ToolsArray(convert.(T, A))
ToolsArray{T}(A::AbstractToolsArray{T}; kw...) where {T} = ToolsArray(A; kw...)

"""
    ToolsArray(f::Function, dim::Dimension; [name])

Apply function `f` across the values of the dimension `dim`
(using `broadcast`), and return the result as a dimensional array with
the given dimension. Optionally provide a name for the result.
"""
function ToolsArray(f::Function, dim::Dimension;
                    name = Symbol(nameof(f), "(", name(dim), ")"), kwargs...)
    ToolsArray(map(f, val(dim)), (dim,); name, kwargs...)
end
function ToolsArray(f::Function, dims::Vararg{Dimension};
                    name = Symbol(nameof(f), "(", join(name.(dims), ','), ")"),
                    kwargs...)
    data = map(Iterators.product(map(val, dims)...)) do args
        f(args...)
    end
    ToolsArray(data, dims; name, kwargs...)
end

## Extra constructors

ToolsArray(x::AbstractArray, D::DimensionalData.Dimension) = ToolsArray(x, (D,))
function ToolsArray(D::DimensionalData.DimArray)
    ToolsArray(D.data, D.dims, D.refdims, D.name, D.metadata)
end

@inline function DimensionalData.rebuild(A::ToolsArray, data::AbstractArray, dims::Tuple,
                                         refdims::Tuple, name, metadata)
    ToolsArray(data, dims, refdims, name, metadata)
end

# * Custom dimensions
import DimensionalData: TimeDim, XDim, YDim, ZDim
DimensionalData.@dim 𝑡 TimeDim "Time"
DimensionalData.@dim 𝑥 XDim "x"
DimensionalData.@dim 𝑧 YDim "y"
DimensionalData.@dim 𝑦 ZDim "z"

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
ToolsDimension = Union{𝑡, 𝑥, 𝑧, 𝑦, 𝑓, Var, Obs, ToolsDim}

function DimensionalData.dimconstructor(::Tuple{ToolsDimension,
                                                Vararg{DimensionalData.Dimension}})
    ToolsArray
end
DimensionalData.dimconstructor(::Tuple{<:ToolsDimension, Vararg}) = ToolsArray
DimensionalData.dimconstructor(dims::ToolsDimension) = ToolsArray

end
