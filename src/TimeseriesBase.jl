module TimeseriesBase

using Reexport
using DimensionalData
using IntervalSets
import DimensionalData: name
@reexport using DimensionalData
@reexport using IntervalSets

include("ToolsArrays.jl")
using TimeseriesBase.ToolsArrays
export AbstractToolsArray, ToolsArray,
       ToolsDimension, ToolsDim,
       𝑡, 𝑥, 𝑦, 𝑧, 𝑓, Var, Obs, Log𝑓, Log10𝑓

include("TimeSeries.jl")
using TimeseriesBase.TimeSeries
export AbstractTimeseries,
       UnivariateTimeseries,
       MultivariateTimeseries,
       RegularTimeseries,
       UnivariateRegular, MultivariateRegular,
       IrregularTimeseries,
       TimeIndex, RegularIndex, RegularTimeIndex,
       IrregularIndex, IrregularTimeIndex,
       Timeseries, Timeseries,
       MultidimensionalIndex, MultidimensionalTimeseries,
       SpikeTrain, MultivariateSpikeTrain, UnivariateSpikeTrain,
       spiketrain, spiketimes

include("Spectra.jl")
using TimeseriesBase.Spectra
export freqs, Spectrum,
       AbstractSpectrum, RegularSpectrum, UnivariateSpectrum, MultivariateSpectrum,
       AbstractSpectrogram, MultivariateSpectrogram, RegularSpectrogram

include("UnitfulTools.jl")
using TimeseriesBase.UnitfulTools
export dimunit, timeunit, frequnit, unit,
       UnitfulIndex, UnitfulTimeseries, UnitfulSpectrum,
       ustripall

include("Utils.jl")
using TimeseriesBase.Utils
export times, step, samplingrate, samplingperiod, duration, coarsegrain,
       buffer, window, delayembed, rectifytime, rectify, matchdim, interlace,
       centraldiff!, centraldiff, centralderiv!, centralderiv,
       rightdiff!, rightdiff, rightderiv!, rightderiv,
       leftdiff!, leftdiff, leftderiv!, leftderiv,
       abs, angle, resultant, resultantlength,
       circularmean, circularvar, circularstd,
       phasegrad, addrefdim, addmetadata, align

include("DatesTools.jl")
using TimeseriesBase.DatesTools
export DateIndex, DateTimeIndex, DateTimeseries

include("Operators.jl")
using TimeseriesBase.Operators
export ℬ, ℬ!, ℒ!, ℒ, 𝒯

include("IO.jl")
using TimeseriesBase.IO
export savetimeseries, savets, loadtimeseries, loadts

end
