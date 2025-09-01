module Spectra

using Statistics

using ..ToolsArrays
using ..TimeSeries
import ..ToolsArrays: FrequencyDim

using DimensionalData

export freqs, Spectrum,
       AbstractSpectrum, RegularSpectrum, UnivariateSpectrum, MultivariateSpectrum,
       AbstractSpectrogram, MultivariateSpectrogram, RegularSpectrogram

"""
    𝑓

A DimensionalData.jl dimension representing the frequency domain.
"""
𝑓

"""
    FreqIndex

A type alias for a tuple of dimensions, where the first dimension is of type `FrequencyDim`.
"""
const FreqIndex = Tuple{A, Vararg{DimensionalData.Dimension}} where {A <: 𝑓}

"""
    AbstractSpectrum{T, N, B}

A type alias for an `AbstractToolsArray` in which the first dimension is [`𝑓`](@ref)requency.
"""
const AbstractSpectrum = AbstractToolsArray{T, N, <:FreqIndex, B} where {T, N, B}
freqs(x::AbstractSpectrum) = lookup(x, 𝑓) |> val

"""
    RegularFreqIndex

A type alias for a tuple of dimensions, where the first dimension is a regularly sampled [`𝑓`](@ref)requency.
"""
const RegularFreqIndex = Tuple{A,
                               Vararg{DimensionalData.Dimension}} where {A <:
                                                                         FrequencyDim{<:RegularIndex}}

"""
    RegularSpectrum{T, N, B}

A type alias for a spectrum with a regularly sampled frequency index.
"""
const RegularSpectrum = AbstractToolsArray{T, N, <:RegularFreqIndex, B} where {T, N, B}

"""
    UnivariateSpectrum{T} = AbstractSpectrum{T, 1} where T

A type alias for a univariate spectrum.
"""
const UnivariateSpectrum = AbstractSpectrum{T, 1} where {T}
"""
    MultivariateSpectrum{T} = AbstractSpectrum{T, 2} where T

A type alias for a multivariate spectrum.
"""
const MultivariateSpectrum = AbstractSpectrum{T, 2} where {T}

"""
    Spectrum(f, x)

Constructs a univariate spectrum with frequencies `f` and data `x`.
"""
Spectrum(f, x; kwargs...) = ToolsArray(x, (𝑓(f),); kwargs...)

"""
    Spectrum(f, v, x)

Constructs a multivariate spectrum with frequencies `f`, variables `v`, and data `x`.
"""
Spectrum(f, v, x; kwargs...) = ToolsArray(x, (𝑓(f), Var(v)); kwargs...)
function Spectrum(f, v::DimensionalData.Dimension, x; kwargs...)
    ToolsArray(x, (𝑓(f), v); kwargs...)
end

import DimensionalData: Dimension, TimeDim

const TimeFreqIndex = Tuple{T, F, Vararg{Dimension}} where {T <: TimeDim, F <: 𝑓}
const RegularTimeFreqIndex = Tuple{T, F,
                                   Vararg{Dimension}} where {
                                                             T <:
                                                             TimeDim{<:RegularIndex},
                                                             F <: 𝑓}

const AbstractSpectrogram = AbstractToolsArray{T, N, <:TimeFreqIndex, B} where {T, N, B}
times(x::AbstractSpectrogram) = lookup(x, 𝑡) |> val
freqs(x::AbstractSpectrogram) = lookup(x, 𝑓) |> val

const MultivariateSpectrogram = AbstractSpectrogram{T, 3} where {T}

const RegularSpectrogram = AbstractToolsArray{T, N, <:RegularTimeFreqIndex,
                                              B} where {T, N, B}

end
