module Utils

using IntervalSets
using DimensionalData
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.UnitfulTools
import TimeseriesBase.UnitfulTools: _unit, _stepquantity
import TimeseriesBase.DatesTools: _diffunit, _offsets, _reinstate, _onoffsets,
    DateIndex
import DimensionalData: Dates

import DimensionalData.Dimensions: At, Near, Dimension
import DimensionalData: print_array, _print_array_ctx, _print_indices_vec
import DimensionalData: At, Between, Touches, Near, Where, Contains
using Unitful
using Statistics

export times, step, samplingrate, samplingperiod, nyquist, duration, coarsegrain, stitch,
    buffer, window, delayembed, rectifytime, rectify, matchdim, regularize,
    interlace,
    centraldiff!, centraldiff, centralderiv!, centralderiv,
    rightdiff!, rightdiff, rightderiv!, rightderiv,
    leftdiff!, leftdiff, leftderiv!, leftderiv,
    abs, angle, resultant, resultantlength,
    circularmean, circularvar, circularstd,
    phasegrad,
    addrefdim, addmetadata, align,
    spiketrain, spiketimes,
    Dropdims

# import LinearAlgebra.mul!
# function mul!(a::AbstractVector, b::AbstractTimeseries, args...; kwargs...)
#     mul!(a, b.data, args...; kwargs...)
# end

Selectors = [:At, :Between, :Touches, :Near, :Where, :Contains]
# Allow dims to be passed directly to selectors
[:($(S)(D::Dimension) = $(S)(parent(val(D)))) for S in Selectors] .|> eval

description(x) = "$(size(x)) $(typeof(x).name.name)"
function print_array(io::IO, mime, A::AbstractDimArray{T, 0}) where {T <: AbstractArray}
    return print(_print_array_ctx(io, T), "\n", description.(A[]))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 1}) where {T <: AbstractArray}
    return Base.print_matrix(_print_array_ctx(io, T), description.(A))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 2}) where {T <: AbstractArray}
    return Base.print_matrix(_print_array_ctx(io, T), description.(A))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 3}) where {T <: AbstractArray}
    i3 = firstindex(A, 3)
    frame = view(A, :, :, i3)
    _print_indices_vec(io, i3)
    return Base.print_matrix(_print_array_ctx(io, T), description.(frame))
end
function print_array(
        io::IO, mime,
        A::AbstractToolsArray{T, N}
    ) where {T <: AbstractArray, N}
    o = ntuple(x -> firstindex(A, x + 2), N - 2)
    frame = view(A, :, :, o...)

    _print_indices_vec(io, o...)
    return Base.print_matrix(_print_array_ctx(io, T), description.(frame))
end

"""
    times(x::AbstractTimeseries)

Returns the time indices of the [`AbstractTimeseries`](@ref) `x`.

## Examples
```@example 1
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, t);
julia> times(ts) == t
```
"""
times(x::AbstractTimeseries) = lookup(x, 𝑡) |> val

"""
    step(x::RegularTimeseries; dims=𝑡)

Returns the step size (time increment) of a regularly sampled [`RegularTimeseries`](@ref).

## Examples
```@example 1
julia> t = 1:100;
julia> x = rand(100);
julia> rts = Timeseries(x, t);
julia> step(rts) == 1
```
"""
Base.step(x::RegularTimeseries; dims = 𝑡) = lookup(x, dims) |> step

"""
    samplingrate(x::RegularTimeseries)

Returns the sampling rate (inverse of the step size) of a regularly sampled [`RegularTimeseries`](@ref).

## Examples
```@example 1
julia> t = 1:100;
julia> x = rand(100);
julia> rts = Timeseries(x, t);
julia> samplingrate(rts) == 1
```

For a `Dates` time index the rate is returned in `Unitful` units (a `Day(1)` step gives
`1/86400 s^-1`). A calendar period (`Year`, `Month`, `Quarter`) has no fixed length, so
no rate exists and an `ArgumentError` is thrown; use [`samplingperiod`](@ref), which
returns the `Period` itself.
"""
samplingrate(x::RegularTimeseries; kwargs...) = 1 / _stepquantity(step(x; kwargs...))

"""
    samplingperiod(x::RegularTimeseries)

Returns the sampling period (step size) of a regularly sampled [`RegularTimeseries`](@ref).

## Examples
```@example 1
julia> t = 1:100;
julia> x = rand(100);
julia> rts = Timeseries(x, t);
julia> samplingperiod(rts) == 1
```
"""
samplingperiod(x::RegularTimeseries; kwargs...) = step(x; kwargs...)

"""
    nyquist(x::RegularTimeseries)

Returns the Nyquist frequency (half the [`samplingrate`](@ref)) of a regularly sampled
[`RegularTimeseries`](@ref): the highest frequency representable without aliasing.

## Examples
```@example 1
julia> t = 0:0.01:10;
julia> x = rand(length(t));
julia> rts = Timeseries(x, t);
julia> nyquist(rts) == samplingrate(rts) / 2
```

Inherits the `Dates` behaviour of [`samplingrate`](@ref).
"""
nyquist(x::RegularTimeseries; kwargs...) = samplingrate(x; kwargs...) / 2

"""
    duration(x::AbstractTimeseries)

Returns the duration of the [`AbstractTimeseries`](@ref) `x`.

## Examples
```@example 1
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, t);
julia> TimeseriesBase.duration(ts) == 99
```
"""
duration(x::AbstractTimeseries) = (last ∘ times)(x) - (first ∘ times)(x)

"""
    IntervalSets.Interval(x::AbstractTimeseries)

Returns an interval representing the range of the [`AbstractTimeseries`](@ref) `x`.

## Examples
```@example 1
julia> using IntervalSets;
julia> t = 1:100;
julia> x = rand(100);
julia> ts = Timeseries(x, t);
julia> IntervalSets.Interval(ts) == (1..100)
```
"""
IntervalSets.Interval(x::AbstractTimeseries) = (first ∘ times)(x) .. (last ∘ times)(x)

"""
    spiketrain(x; kwargs...)

Construct a [`SpikeTrain`](@ref) from a vector of spike times `x`.

The input vector `x` is sorted and converted into a binary time series where each time point
corresponds to a spike (value of `true`).

# Arguments
- `x`: A vector of spike times (will be sorted).
- `kwargs...`: Additional keyword arguments passed to the [`Timeseries`](@ref) constructor.

# Returns
- A [`SpikeTrain`](@ref) (binary time series) with `true` values at the sorted spike times.

See also: [`spiketimes`](@ref)

# Examples
```julia
julia> spike_times = [1.5, 3.2, 0.8, 5.1];
julia> st = spiketrain(spike_times);
julia> times(st)  # Returns sorted spike times: [0.8, 1.5, 3.2, 5.1]
```
"""
function spiketrain(x; kwargs...)
    return Timeseries(trues(length(x)), sort(x); kwargs...)
end

"""
    spiketimes(x::UnivariateSpikeTrain)
    spiketimes(x::SpikeTrain)
    spiketimes(x::AbstractArray)

Extract spike times from a [`SpikeTrain`](@ref) or pass through an array unchanged.

For a univariate spike train, returns the time indices where spikes occur (where the value is `true`).
For a multivariate spike train, returns an array where each element contains the spike times for
one channel/dimension. For a plain array, returns the array unchanged (identity function).

# Arguments
- `x`: A [`SpikeTrain`](@ref) or [`AbstractArray`](@ref).

# Returns
- For [`UnivariateSpikeTrain`](@ref): A vector of spike times.
- For multivariate [`SpikeTrain`](@ref): An array of spike time vectors, one per channel.
- For [`AbstractArray`](@ref): The input array unchanged.

See also: [`spiketrain`](@ref), [`times`](@ref)

# Examples
```julia
julia> spike_times = [1.0, 2.5, 4.0];
julia> st = spiketrain(spike_times);
julia> spiketimes(st)  # Returns [1.0, 2.5, 4.0]
```
"""
function spiketimes(x::UnivariateSpikeTrain)
    return times(x[x])
end
function spiketimes(x::SpikeTrain)
    return map(spiketimes, eachslice(x, dims = tuple(2:ndims(x)...)))
end
spiketimes(x::AbstractArray) = x

"""
    interlace(x::UnivariateTimeseries, y::UnivariateTimeseries)

Interleave two univariate time series into a single series whose time index is the sorted
union of `times(x)` and `times(y)`, with the values reordered to match.
"""
function interlace(x::UnivariateTimeseries, y::UnivariateTimeseries)
    ts = vcat(times(x), times(y))
    idxs = sortperm(ts)
    ts = ts[idxs]
    data = vcat(x.data, y.data)
    data = data[idxs]
    return Timeseries(data, ts)
end

function _buffer(x, n::Integer, p::Integer = 0; discard::Bool = true)
    y = [@views x[i:min(i + n - 1, end)] for i in 1:(n - p):length(x)]
    while discard && !isempty(y) && length(y[end]) < n
        pop!(y)
    end
    return y
end
function _buffer(x::AbstractMatrix, n::Integer, p::Integer = 0; discard::Bool = true)
    y = [@views x[i:min(i + n - 1, end), :] for i in 1:(n - p):size(x, 1)]
    while discard && !isempty(y) && size(y[end], 1) < n
        pop!(y)
    end
    return y
end
buffer(x::AbstractVector, args...; kwargs...) = _buffer(x, args...; kwargs...)

"""
    buffer(x::RegularTimeseries, n::Integer, p::Integer; kwargs...)

Buffer a time series `x` with a given window length and overlap between successive buffers.

For a `Dates` time index each buffer is labelled with the mean of its own time values,
computed on integer offsets since instants do not add.

## Arguments
- `x`: The regular time series to be buffered.
- `n`: The number of samples in each buffer.
- `p`: The number of samples of overlap betweeen the buffers.
    - `0` indicates no overlap
    - +`2` indicates `2` samples of overlap between successive buffers
    - -`2` indicates `2` samples of gap between buffers

See also: [`window`](@ref), [`delayembed`](@ref), [`coarsegrain`](@ref)
"""
function buffer(x::RegularTimeseries, args...; kwargs...)
    y = _buffer(x, args...; kwargs...)
    # No complete buffers (e.g. window longer than the series): return an empty
    # time series rather than indexing into an empty buffer list.
    # An explicitly empty lookup, not `times(x)[1:0]`: indexing a `Period`-stepped range
    # with an empty range needs `copysign(::Day, ::Day)`, which Base does not define.
    isempty(y) && return Timeseries(y, eltype(times(x))[])
    tb = _buffer(times(x), args...; kwargs...)
    # Instants do not add and periods do not divide exactly, so centres and the grid
    # below both go through `_onoffsets`; this is also why the three-argument `range`
    # is applied to offsets, since `range(::DateTime, ::DateTime, n)` yields raw
    # millisecond counts rather than instants.
    t = _onoffsets.(mean, tb)
    # Buffer centres are regular only when every buffer is full; a short final buffer
    # (`discard = false`) sits off that grid, so label it with its own centre.
    ts = allequal(length.(tb)) ?
        _onoffsets(o -> range(first(o), last(o), length(y)), t) : t
    return y = Timeseries(y, ts)
end

"""
    window(x::RegularTimeseries, n::Integer, p::Integer; kwargs...)

Window a time series `x` with a given window length and step between successive windows.

## Arguments
- `x`: The regular time series to be windows.
- `n`: The number of samples in each window.
- `p`: The number of samples to slide each successive window.

See also: [`buffer`](@ref), [`delayembed`](@ref), [`coarsegrain`](@ref)
"""
window(x, n, p = n, args...; kwargs...) = buffer(x, n, n - p, args...; kwargs...)

function _delayembed(x::AbstractVector, n, τ, p = 1; kwargs...) # A delay embedding with dimension `n`, delay `τ`, and skip length of `p`
    y = window(x, n * τ, p; kwargs...)
    return y = map(y) do _y
        @view _y[1:τ:end]
    end
end
delayembed(x::AbstractVector, args...; kwargs...) = _delayembed(x, args...; kwargs...)

"""
    delayembed(x::UnivariateRegular, n::Integer, τ::Integer, p::Integer=1; kwargs...)

Delay embed a univariate time series `x` with a given dimension `n`, delay `τ`, and skip length of `p`

## Arguments
- `x`: The regular time series to be delay embedded.
- `n`: The embedding dimension, i.e., the number of samples in each embedded vector.
- `τ`: The number of original sampling periods between each sample in the embedded vectors.
- `p`: The number of samples to skip between each successive embedded vector.

See also: [`buffer`](@ref), [`window`](@ref)
"""
function delayembed(x::UnivariateRegular, n, τ, p = 1, args...; kwargs...)
    y = _delayembed(x, n, τ, p, args...; kwargs...)
    ts = last.(times.(y))  # Time of the head of the vector
    dt = step(x) * p
    ts = ts[1]:dt:(ts[1] + dt * (length(y) - 1))
    δt = τ * step(x) # spacing *within* an embedded vector; the skip `p` is between them
    # `zero(δt)`, not `0`: a `Period`-stepped range needs a `Period` endpoint.
    delays = (-(δt * (n - 1))):δt:Base.zero(δt)
    y = set.(y, [𝑡 => Dim{:delay}(delays)])
    y = set(y, 𝑡 => ts) # Set time index to start time of each time series
    return y = stack(y, dims = 1) # dims=1 so time is on first dimension
end

# ============================================================================
# regularize: unified replacement for rectify / rectifytime / matchdim.
#
# One operation, three methods:
#
#   regularize(d::Dimension; ...)                  → (new_lookup, orig_lookup)
#   regularize(X::AbstractDimArray; dims=𝑡, ...)   → X with regular lookup(s)
#   regularize(Xs::AbstractVector{<:AbstractDimArray}; dims=𝑡, ...)
#       regularize(X1, X2, ...; dims=𝑡, ...)        → vector of arrays sharing
#                                                     a single regular grid
#
# Tolerance semantics (only one knob, with no silent rescaling):
#   - `atol`: max allowed absolute deviation of any input lookup point from the
#     best-fit regular grid. Used as both the regularity check and the rounding
#     precision. Carries the dim's units. Default `1e-6 * mean_step`.
#   - `sigdigits`: optional override for the number of significant digits used
#     to round the step / endpoints. Default: derived from `atol` so rounding
#     never coarsens beyond the tolerance.
#   - `strict=true` (default): throw on regularity failure rather than warn and
#     pass through silently. Set `strict=false` to recover the old behaviour.
#
# Differences from the legacy trio:
#   - The grid is built directly with `range(start, step, length)` — no
#     "extend by 10000*stp then trim" trick.
#   - The regularity check is `maximum(abs(t - best_fit))`, which catches drift
#     and isolated gaps that std-of-diffs misses; the warning names the worst
#     offender.
#   - `zero=true` stores the *genuine* original lookup in metadata.
#   - `regularize(X1, X2, ...)` is the single "align to common grid" path used
#     by both the old vararg `rectify` and `matchdim`.
# ============================================================================

# Fit `t[i] ≈ a + b·(i-1)` by ordinary least squares over `i = 1..n`. Closed
# form, O(n), no allocations beyond the input. Distributes residual evenly
# across all samples (unlike endpoint-pinning, which forces error onto the
# interior and is sensitive to a noisy first/last sample).
#
# Returns `(start, step)` in the bare (unit-stripped) numeric type.
function _lsq_regular_fit(tsbare::AbstractVector)
    n = length(tsbare)
    # sum_{i=0..n-1} i  = n(n-1)/2;  sum i² = n(n-1)(2n-1)/6
    # Centred form keeps the denominator exact and avoids cancellation.
    ibar = (n - 1) / 2
    tbar = sum(tsbare) / n
    num = zero(eltype(tsbare))
    den = zero(typeof(ibar))
    @inbounds for i in 1:n
        di = (i - 1) - ibar
        num += di * (tsbare[i] - tbar)
        den += di * di
    end
    b = num / den          # step
    a = tbar - b * ibar    # start (value at i=1, since ibar is offset from i=1)
    return a, b
end

# Round `x` to the decimal place implied by the quantum `q` (the rounding
# precision). We deliberately round to a decimal *place* rather than to a
# literal multiple of `q`: `round(x/q)*q` reintroduces float error (neither
# `x/q`, `q`, nor their product is exact), so even a clean `0.1` would come
# back as `0.10000000000000002`. `round(x; digits)` instead returns the
# closest `Float64` to the base-10 rounded value, so decimal-friendly steps
# (0.1, 0.25, 1e6, …) snap back exactly. Still scale-invariant: the decimal
# place tracks the magnitude of `q`, which itself tracks the step/atol.
_round_to(x, q) = q > 0 ? round(x; digits = -floor(Int, log10(q))) : x

# As `_round_to`, but to the *nearest* decimal place rather than the one at or below
# `q`. Flooring can spend only a tenth of the budget it is given (`q = 0.099` rounds at
# `0.01`), which is the difference between snapping jitter off a step and leaving it.
_round_near(x, q) = q > 0 ? round(x; digits = -round(Int, log10(q))) : x

# Bare-bones lookup → (regular_range, orig_values, max_dev, worst_idx).
# `orig` is the untouched input collected to a vector so callers that want to
# preserve the raw lookup (e.g. `zero=true`) have it.
function _fit_regular_grid(
        ts::AbstractVector; atol = nothing, sigdigits = nothing,
        quantum = nothing
    )
    n = length(ts)
    n < 2 && throw(ArgumentError("regularize: need at least 2 lookup points, got $n"))
    u = _unit(eltype(ts))
    tsbare = ustripall(ts)

    # Fast path: input is already a regular range. No fit needed, no rounding
    # error introduced, and we can report exactly zero deviation.
    if ts isa AbstractRange && step(ts) isa Number
        grid_zero_dev = u == NoUnits ? zero(eltype(tsbare)) : zero(eltype(tsbare)) * u
        return ts, collect(ts), grid_zero_dev, 1
    end

    issorted(tsbare) ||
        throw(ArgumentError("regularize: input lookup is not sorted; sort it first"))

    t0bare, stpbare = _lsq_regular_fit(tsbare)

    # Rounding precision. The legacy code rounded to a fixed decimal-digit
    # count, which silently couples to the magnitude of the step. We instead
    # round start and step to the decimal place implied by `q` (see
    # `_round_to`), so the precision scales with the data.
    #
    # Start and step get different budgets, because their errors propagate
    # differently: a start offset shifts every sample equally (`atol/10` is
    # already well inside tolerance), while a step offset shifts sample `i` by
    # `i·δ`, so it compounds and needs `atol/2n` to keep the *last* sample
    # inside `atol`. Using the start's coarser quantum for both is what let a
    # long grid drift out of tolerance; using the step's finer quantum for both
    # is too fine to snap jitter off the start. Worst case here is
    # `atol/20 + atol/4`, comfortably inside `atol`.
    #
    # `atol` is converted to the lookup's own unit first: stripping it
    # unconverted makes `100u"ms"` mean `100` on a lookup in seconds.
    # `sigdigits` is a manual override and bypasses both budgets.
    atolbare = atol === nothing ? nothing :
        atol isa Quantity ? ustrip(u, atol) : ustripall(atol)
    t0_r, stp_r = if quantum !== nothing
        # ponytail: `quantum` exists solely for the Dates path, whose lookup is
        # integer-quantised and so cannot use the decimal-place rounding the numeric
        # path is built around (a loose `atol` would round 86_400_000 ms to
        # 100_000_000). Drop it if a second caller never appears.
        round(t0bare / quantum) * quantum, round(stpbare / quantum) * quantum
    elseif sigdigits !== nothing
        round(t0bare; sigdigits), round(stpbare; sigdigits)
    else
        # No tolerance given: fall back to `default_atol = step·1e-6`
        # (see `_default_atol`), which lets near-integer / near-rational
        # lookups snap back to clean values.
        a = atolbare !== nothing && atolbare > 0 ? atolbare :
            ustripall(_default_atol(stpbare))
        _round_to(t0bare, a / 10), _round_near(stpbare, a / (2 * n))
    end

    grid_bare = range(start = t0_r, step = stp_r, length = n)
    # Measure deviation against the grid we actually return, not the unrounded
    # fit; otherwise the check certifies a grid nobody receives.
    maxdev, worst = findmax(abs.(tsbare .- grid_bare))
    grid = u == NoUnits ? grid_bare : grid_bare .* u
    maxdev_u = u == NoUnits ? maxdev : maxdev * u
    return grid, collect(ts), maxdev_u, worst
end

# Default tolerance: 1e-6 of the step magnitude. Tight enough to catch real
# irregularity, loose enough to absorb the float jitter that motivates calling
# this in the first place. Caller should pass the LSQ-fitted step (with units)
# so we don't re-fit.
_default_atol(step_with_units) = 1.0e-6 * abs(step_with_units)

# The lookup `regularize(...; zero = true)` applies: elapsed from the origin. On an
# instant axis that is a *duration* range, and it must be built in a single unit:
# mixing `Base.zero(::DateTime) === Millisecond(0)` with a `Day(1)` step gives a
# `CompoundPeriod` range that DimensionalData cannot order.
_zerogrid(grid) = range(
    start = Base.zero(first(grid)), step = step(grid),
    length = length(grid)
)
function _zerogrid(grid::AbstractRange{<:Dates.TimeType})
    P = _diffunit(eltype(grid))
    return range(P(0), step = convert(P, step(grid)), length = length(grid))
end

# Normalise a `dims` argument (a single dim, a `Tuple`, or a vector) to a `Vector` of dims.
_dimlist(dims) = dims isa Tuple || dims isa AbstractVector ? collect(dims) : [dims]

function _check_regularity(maxdev, worst, ts, dim, atol, fitted_step; strict = true)
    tol = atol === nothing ? _default_atol(fitted_step) : atol
    if maxdev > tol
        msg = "regularize: lookup along $dim is not regular within atol=$tol " *
            "(max deviation $maxdev at index $worst, value $(ts[worst]))"
        strict ? throw(ArgumentError(msg)) : @warn msg
        return false
    end
    return true
end

"""
    regularize(d::DimensionalData.Dimension; atol=nothing, sigdigits=nothing,
               strict=true)

Return `(new_lookup, original_lookup)` where `new_lookup` is a regular `range`
that best fits the values of `d` and `original_lookup` is the input lookup
collected to a vector.

Use this method when you want to inspect the rectified grid yourself; most
callers should use [`regularize`](@ref) on an `AbstractDimArray` instead.

# Keyword arguments
- `atol`: maximum allowed absolute deviation of any lookup point from the
  returned grid. Converted to the lookup's own units, so `100u"ms"` and
  `0.1u"s"` mean the same thing on a lookup in seconds. Defaults to
  `1e-6 * abs(step)` (see `_default_atol`).
- `sigdigits`: override the number of digits used to round the step and
  start. Defaults to a value derived from `atol` so rounding error stays
  within tolerance.
- `strict=true`: throw if regularity fails. Set `false` to warn and return the
  best-fit grid anyway.
"""
function regularize(
        d::DimensionalData.Dimension; atol = nothing, sigdigits = nothing,
        strict = true, dim_label = nameof(typeof(d))
    )
    ts = collect(d)
    grid, orig, maxdev, worst = _fit_regular_grid(ts; atol, sigdigits)
    _check_regularity(maxdev, worst, ts, dim_label, atol, step(grid); strict)
    return grid, orig
end

"""
    regularize(d::DimensionalData.Dimension{<:DateIndex}; kwargs...)

As above, for a `Dates` lookup. The fit runs on exact `Int64` offsets from the first
value and rounds to whole counts of the lookup's resolution (milliseconds for
`DateTime`, days for `Date`), because rounding to a decimal place is meaningless on
integers.

`atol` may be given as a `Period`, or as a bare count of that resolution. It defaults
to *exact*, not to `1e-6 * step`: a `Dates` lookup carries no float jitter, so any
deviation from the fit is genuine irregularity rather than noise. A fitted step below
one whole unit is rejected, since `DateTime` silently rounds anything finer.
"""
function regularize(
        d::DimensionalData.Dimension{<:DateIndex}; atol = nothing,
        sigdigits = nothing, strict = true, dim_label = nameof(typeof(d))
    )
    lk = parent(DimensionalData.lookup(d))
    P = _diffunit(eltype(lk))
    # A Dates range is exact by construction, so there is nothing to repair and no
    # reason to relabel its step (a `Day(1)` step should not come back as
    # `Millisecond(86400000)`).
    lk isa AbstractRange && return lk, collect(lk)
    t0 = first(lk)
    tol = atol === nothing ? 0 :
        atol isa Dates.Period ? Dates.value(convert(P, atol)) : atol
    grid_o, _, maxdev, worst = _fit_regular_grid(
        _offsets(lk, t0); atol = tol, sigdigits, quantum = 1
    )
    abs(step(grid_o)) < 1 && throw(
        ArgumentError(
            "regularize: fitted step $(step(grid_o)) is below one $(nameof(P)), the " *
                "resolution of a $(eltype(lk)) lookup; use a finer lookup type"
        )
    )
    grid, orig = _reinstate(t0, grid_o), collect(lk)
    _check_regularity(
        P(round(Int64, maxdev)), worst, orig, dim_label, P(tol), step(grid); strict
    )
    return grid, orig
end

"""
    regularize(X::AbstractDimArray; dims=𝑡, atol=nothing, sigdigits=nothing,
               zero=false, strict=true)

Replace the lookup of `X` along each of `dims` with a regular range that best
fits the existing values, repairing accumulated float jitter. If a lookup
deviates from regular by more than `atol` an `ArgumentError` is thrown
(`strict=false` downgrades this to a warning).

If `zero=true`, the new lookup starts at zero and the genuine original
lookup is stored in `metadata(X)` under the dimension name. On a `Dates` lookup
`zero=true` means elapsed time, so it returns a uniform-unit `Period` range rather
than instants.

A `Dates` lookup is fitted on exact integer offsets; see the `Dimension` method above
for the tolerance rules, which differ from the numeric ones.

This replaces the older `rectify` and `rectifytime` methods.
"""
function regularize(
        X::AbstractDimArray; dims = 𝑡, atol = nothing, sigdigits = nothing,
        zero = false, strict = true
    )
    dimlist = _dimlist(dims)
    for dim in dimlist
        d = DimensionalData.dims(X, dim)
        grid, orig = regularize(
            d; atol, sigdigits, strict,
            dim_label = DimensionalData.name(d)
        )
        new_grid = zero ? _zerogrid(grid) : grid
        X = set(X, dim => parent(new_grid))
        if zero
            X = rebuild(
                X;
                metadata = Dict(
                    pairs(metadata(X))...,
                    Symbol(DimensionalData.name(d)) => orig,
                )
            )
        end
    end
    return X
end

"""
    regularize(Xs::AbstractVector{<:AbstractDimArray}; dims=𝑡, atol=nothing,
               sigdigits=nothing, zero=false, strict=true)
    regularize(X1, X2, ...; dims=𝑡, kwargs...)

Align a collection of arrays to a common regular grid along each of `dims`.

The shared grid is computed from the element-wise mean of the lookups *after*
each array has been cropped to the maximal common range and trimmed to the
minimum common length. The same regularity check that the single-array method
uses then applies to every input array — if any one of them deviates from the
common grid by more than `atol`, an `ArgumentError` is thrown (or a warning,
under `strict=false`) naming the offending array and index.

This replaces the older `matchdim` and the vararg form of `rectify`.
"""
function regularize(
        Xs::AbstractVector{<:AbstractDimArray}; dims = 𝑡, atol = nothing,
        sigdigits = nothing, zero = false, strict = true
    )
    isempty(Xs) && return Xs
    dimlist = _dimlist(dims)

    for dim in dimlist
        # A Dates lookup has no mean, no median and no exact halving; its integer
        # offsets have all three. Swap the whole dimension over, let the numeric body
        # below run unchanged, and map back once at the end. The origin is shared
        # across arrays so the offsets stay comparable.
        lk1 = parent(lookup(Xs[1], dim))
        isdates = eltype(lk1) <: Dates.AbstractTime
        t0 = isdates ? first(lk1) : nothing
        P = isdates ? _diffunit(eltype(lk1)) : nothing
        # A Dates lookup carries no float jitter, so the numeric default (which exists
        # to absorb it) would be too loose: the default here is exact.
        _atol = if !isdates
            atol
        elseif atol === nothing
            0
        elseif atol isa Dates.Period
            Dates.value(convert(P, atol))
        else
            atol
        end
        _quantum = isdates ? 1 : nothing
        _show(v) = isdates ? _reinstate(t0, v) : v
        if isdates
            Xs = [
                set(x, dim => _offsets(parent(lookup(x, dim)), t0)) for x in Xs
            ]
        end

        all_dims = [DimensionalData.dims(x, dim) for x in Xs]
        # Crop to the maximal common range, then trim to the minimum common length.
        mint = maximum(minimum(d) for d in all_dims)
        maxt = minimum(maximum(d) for d in all_dims)
        mint > maxt &&
            throw(ArgumentError("regularize: no overlapping range along $dim"))
        # Pad the crop interval by half a sampling step so that boundary points
        # sitting within float jitter of the overlap edge are kept in *every*
        # array, not dropped from some — otherwise the index-based trim below
        # would misalign the arrays. A genuinely different range (≥ one step
        # apart) is still cropped correctly.
        steps = [
            median(abs.(diff(collect(parent(lookup(x, dim)))))) for x in Xs
                if size(x, dim) > 1
        ]
        pad = isempty(steps) ? Base.zero(mint) : minimum(steps) / 2
        Xs = [
            x[rebuild(DimensionalData.dims(x, dim), (mint - pad) .. (maxt + pad))]
                for x in Xs
        ]
        L = minimum(size(x, dim) for x in Xs)
        Xs = [selectdim(x, dimnum(x, dim), 1:L) for x in Xs]

        # Common grid from the element-wise mean of the (cropped, trimmed)
        # lookups. Pull the bare lookup *values* (not `collect(::Dimension)`,
        # which here yields a ToolsArray and would make `mean` try to add
        # arrays with mismatched lookups).
        lookups = [collect(parent(lookup(x, dim))) for x in Xs]
        mean_lookup = mean(lookups)
        grid, _ = _fit_regular_grid(
            mean_lookup; atol = _atol, sigdigits,
            quantum = _quantum
        )
        isdates && abs(step(grid)) < 1 && throw(
            ArgumentError(
                "regularize: fitted step is below one $(nameof(P)), the resolution " *
                    "of a $(eltype(lk1)) lookup; use a finer lookup type"
            )
        )

        # Verify every input is within tolerance of the common grid and
        # report the worst offender by array index.
        dim_label = DimensionalData.name(DimensionalData.dims(Xs[1], dim))
        tol = _atol === nothing ? _default_atol(step(grid)) : _atol
        for (i, xl) in enumerate(lookups)
            devs = abs.(xl .- grid)
            maxdev, worst = findmax(devs)
            if maxdev > tol
                msg = "regularize: array $i lookup along $dim_label deviates " *
                    "from common grid by $(isdates ? P(round(Int64, maxdev)) : maxdev) > " *
                    "atol=$(isdates ? P(tol) : tol) " *
                    "(worst at index $worst, value $(_show(xl[worst])))"
                strict ? throw(ArgumentError(msg)) : @warn msg
            end
        end

        # Apply the (possibly zero-shifted) grid and stash the *genuine*
        # original lookup (captured before `set`). `zero = true` means elapsed from
        # the origin, so on a Dates axis it deliberately yields a uniform-unit
        # `Period` range rather than instants.
        new_grid = if zero
            isdates ?
                range(
                    P(0), step = P(round(Int64, step(grid))),
                    length = length(grid)
                ) :
                _zerogrid(grid)
        else
            isdates ? _reinstate(t0, grid) : grid
        end
        Xs = map(zip(Xs, lookups)) do (x, origlk)
            x = set(x, dim => parent(new_grid))
            zero ?
                rebuild(
                    x;
                    metadata = Dict(
                        pairs(metadata(x))...,
                        Symbol(dim_label) => _show(origlk),
                    )
                ) :
                x
        end
    end
    return Xs
end

regularize(X1::AbstractDimArray, Xrest::AbstractDimArray...; kwargs...) =
    regularize(AbstractDimArray[X1, Xrest...]; kwargs...)

"""
    rectify(ts::Dimension; tol=4, zero=false)
    rectify(X::AbstractDimArray; dims, tol=4, zero=false)
    rectify(X1, X2, ...; dims=𝑡, tol=4, zero=false)

Replace a near-regular lookup with a regular range, rounding the step to `tol` significant
figures. A `Dates` lookup is rejected: significant figures of a decimal step have no
meaning on integer counts, so use [`regularize`](@ref) instead. The array forms operate along `dims`; the vararg form additionally aligns several
arrays onto a shared grid. With `zero=true` the lookup starts at zero and the original is
stored in metadata.

!!! note
    Superseded by [`regularize`](@ref), which fits the grid by least squares, checks
    regularity by maximum deviation, and throws (rather than warns) on failure by default.
    Prefer `regularize` for new code.
"""
function rectify(
        ts::DimensionalData.Dimension; tol = 4, zero = false, extend = false,
        atol = nothing
    )
    # A Dates lookup is rejected rather than ported: this path's tolerance model is
    # significant figures of a decimal step, which has no meaning on integer counts.
    eltype(ts) <: Dates.AbstractTime && throw(
        ArgumentError(
            "rectify: a Dates lookup is not supported; use `regularize`, which fits " *
                "the grid on integer offsets and takes `atol` as a Period"
        )
    )
    u = _unit(eltype(ts))
    ts = collect(ts)
    origts = ts
    stp = ts |> diff |> mean
    err = ts |> diff |> std
    tol = Int(tol - round(log10(abs(stp |> ustripall))))

    if isnothing(atol) && ustripall(err) > exp10(-tol - 1)
        @warn "Step $stp is not approximately constant (err=$err, tol=$(exp10(-tol - 1))), skipping rectification"
    else
        if !isnothing(atol)
            tol = atol
        end
        stp = u == NoUnits ? round(stp; digits = tol) : round(u, stp; digits = tol)
        # `first`/`last`, not `extrema`: a descending lookup must keep its direction.
        ends = (first(ts), last(ts))
        t0, t1 = u == NoUnits ? round.(ends; digits = tol) :
            round.(u, ends; digits = tol)
        if zero
            origts = t0:stp:(t1 + (10000 * stp))
            t1 = t1 - t0
            t0 = 0
        end
        if extend
            ts = t0:stp:(t1 + (10000 * stp))
        else
            ts = range(start = t0, step = stp, length = length(ts))
        end
    end
    return parent(ts), origts
end

function rectify(X::AbstractDimArray; dims, tol = 4, zero = false, kwargs...) # tol gives significant figures for rounding
    for dim in _dimlist(dims)
        ts, origts = rectify(
            DimensionalData.dims(X, dim); tol, zero, extend = true,
            kwargs...
        )
        ts = ts[1:size(X, dim)] # Should be ok?
        @assert length(ts) == size(X, dim)
        X = set(X, dim => ts)
        @assert lookup(X, dim) == ts
        if zero
            X = rebuild(X; metadata = (Symbol(dim) => origts, pairs(metadata(X))...))
        end
    end
    return X
end

function rectify(
        X::Vararg{AbstractDimArray}; dims = 𝑡, tol = 4, zero = false,
        kwargs...
    )

    # Process each dimension
    for dim in _dimlist(dims)
        # Normalise once: `dim` may arrive as an index, a type or a dimension, and only a
        # type is callable, which several uses below relied on.
        dim = DimensionalData.dims(first(X), dim)
        # As the single-array method: a Dates lookup goes to `regularize`. Guarded here
        # too, since this path averages the lookups before reaching that method.
        eltype(lookup(first(X), dim)) <: Dates.AbstractTime && throw(
            ArgumentError(
                "rectify: a Dates lookup is not supported; use " *
                    "`regularize(Xs; dims)`, which aligns arrays on a common grid"
            )
        )
        # Extract dimension values from all arrays
        all_dims = [DimensionalData.dims(x, dim) for x in X]

        # Find common range across all arrays
        mint = maximum([minimum(d) for d in all_dims])
        maxt = minimum([maximum(d) for d in all_dims])

        # Check if there's overlap
        if mint > maxt
            @error "No overlapping range found for dimension $dim"
            return X
        end

        # Subset all arrays to common range (with small tolerance for floating point)
        u = _unit(eltype(all_dims[1]))
        tol_val = u == NoUnits ? exp10(-tol) : exp10(-tol) * u
        common_range = (mint - tol_val) .. (maxt + tol_val)
        X = [x[rebuild(DimensionalData.dims(x, dim), common_range)] for x in X]

        # Find minimum common length
        min_length = minimum([size(x, dim) for x in X])
        X = [selectdim(x, dimnum(x, dim), 1:min_length) for x in X]

        # Compute mean dimension values for rectification
        mean_dim_vals = mean([collect(DimensionalData.dims(x, dim)) for x in X])

        # Rectify using the mean dimension values
        ts, origts = rectify(
            rebuild(DimensionalData.dims(X[1], dim), mean_dim_vals);
            tol = tol, zero = zero, extend = true, kwargs...
        )

        # Trim to actual size
        ts = ts[1:min_length]

        # Verify rectification is within tolerance
        for (i, x) in enumerate(X)
            x_dims = collect(DimensionalData.dims(x, dim))
            max_diff = maximum(abs.(ts .- x_dims))
            # Base tolerance on rectification precision, not step variability
            u = _unit(eltype(x_dims))
            expected_tol = u == NoUnits ? exp10(-tol) : exp10(-tol) * u

            if max_diff > expected_tol
                @warn "Array $i: dimension $dim differs from rectified values by up to $max_diff (tolerance: $expected_tol)"
            end
        end

        # Apply rectified dimension to all arrays
        X = [set(x, dim => ts) for x in X]

        # Add original dimension values to metadata if zero=true
        if zero
            X = [
                rebuild(
                        x;
                        metadata = (
                            Symbol(DimensionalData.name(dim)) => origts[1:min_length],
                            pairs(metadata(x))...,
                        )
                    ) for x in X
            ]
        end
    end

    return X
end

rectifytime(ts::𝑡; kwargs...) = rectify(ts; kwargs...)

"""
    rectifytime(X::AbstractTimeseries; tol = 6, zero = false)

Rectifies the time values of an [`IrregularTimeseries`](@ref). Checks if the time step of
the input time series is approximately constant. If it is, the function rounds the time step
and constructs a [`RegularTimeseries`](@ref) with range time indices. If the time step is
not approximately constant, a warning is issued and the rectification is skipped.

# Arguments
- `X::IrregularTimeseries`: The input time series.
- `tol::Int`: The number of significant figures for rounding the time step. Default is 6.
- `zero::Bool`: If `true`, the rectified time values will start from zero. Default is
  `false`.
"""
rectifytime(X::Vararg{AbstractTimeseries}; kwargs...) = rectify(X...; dims = 𝑡, kwargs...)

"""
    matchdim(X::AbstractVector{<:AbstractDimArray}; dims=1, tol=4, zero=false)

Align a collection of dimensional arrays onto a common rectified grid along `dims`, so that
every element of `X` shares an identical lookup. A `Dates` lookup is rejected; use
`regularize(Xs; dims)`.

!!! note
    Superseded by [`regularize`](@ref), which fits the grid by least squares and applies a
    stricter regularity check; prefer `regularize` for new code.
"""
function matchdim(
        X::AbstractVector{<:AbstractDimArray}; dims = 1, tol = 4, zero = false,
        kwargs...
    )
    # Generate some common time indices as close as possible to the rectified times of each element of the input vector. At most this will change each time index by a maximum of 1 sampling period. We could do better--maximum of a half-- but leave that for now.
    # As `rectify`: a Dates lookup goes to `regularize`, which aligns several arrays on
    # a common grid without inventing decimal-tolerance semantics for integer counts.
    eltype(lookup(X |> first, dims)) <: Dates.AbstractTime && throw(
        ArgumentError(
            "matchdim: a Dates lookup is not supported; use " *
                "`regularize(Xs; dims)`, which aligns arrays on a common grid"
        )
    )
    u = lookup(X |> first, dims) |> eltype |> _unit
    ts = lookup.(X, [dims])
    mint = (maximum(minimum.(ts)) - exp10(-tol) * u) ..
        (minimum(maximum.(ts)) + exp10(-tol) * u)
    X = map(X) do x
        d = rebuild(DimensionalData.dims(x, dims), mint)
        x = getindex(x, d)
    end
    L = minimum(size.(X, dims))
    X = map(X) do x
        d = rebuild(DimensionalData.dims(x, dims), 1:L)
        x = getindex(x, d) # Should now have same length for all inputs
    end

    ts = mean(lookup.(X, [dims]))
    ts, origts = rectify(
        rebuild(DimensionalData.dims(X[1], dims), ts); tol, zero,
        kwargs...
    )
    if any([any(ts .- lookup(x, dims) .> std(ts) / exp10(-tol)) for x in X])
        @error "Cannot find common dimension indices within tolerance"
    end
    X = [set(x, rebuild(DimensionalData.dims(x, dims), ts)) for x in X]
    return X
end

"""
    phasegrad(x, y)

The signed circular difference between angles `x` and `y` (in radians), wrapped to
`[-π, π)`: the shortest signed rotation from `y` to `x`. For example
`phasegrad(0.1, 2π - 0.1) ≈ 0.2`, not `≈ -2π`. Broadcasts over arrays, and uses `angle`
for `Complex` inputs.
"""
phasegrad(x::Real, y::Real) = mod(x - y + π, 2π) - π # +pi - pi because we want the difference mapped from -pi to +pi, so we can represent negative changes.
phasegrad(x, y) = phasegrad.(x, y)
phasegrad(x::Complex, y::Complex) = phasegrad(angle(x), angle(y))

function _centraldiff!(x; grad = -, dims = nothing) # Dims unused
    # a = x[2] # Save here, otherwise they get mutated before we use them
    # b = x[end - 1]
    if grad == -
        x[2:(end - 1)] .= grad(x[3:end], x[1:(end - 2)]) / 2
    else # For a non-euclidean metric, we need to calculate both sides individually
        x[2:(end - 1)] .= (
            grad(x[3:end], x[2:(end - 1)]) +
                grad(x[2:(end - 1)], x[1:(end - 2)])
        ) / 2
    end
    # x[[1, end]] .= [grad(a, x[1]), grad(x[end], b)]
    x[[1, end]] .= [copy(x[2]), copy(x[end - 1])]
    return nothing
end

_diff!(x::UnivariateRegular, f!; kwargs...) = f!(parent(x); kwargs...)
function _diff!(x::AbstractDimArray, f!; dims = 1, kwargs...)
    if !(DimensionalData.lookup(x, dims).data isa AbstractRange)
        error("Differencing dimension must be regularly sampled")
    end
    p = parent(x)
    ndims(p) == 1 && return f!(p; kwargs...)
    # The kernels index linearly, so each 1D slice along `dims` must be differenced on its
    # own; handing them the whole parent leaks the boundary rule across slice boundaries.
    d = DimensionalData.dimnum(x, dims)
    for v in eachslice(p; dims = ntuple(i -> i < d ? i : i + 1, ndims(p) - 1))
        f!(v; kwargs...)
    end
    return nothing
end

"""
    centraldiff!(x::RegularTimeseries; dims=𝑡, grad=-)

Compute the central difference of a regular time series `x`, in-place.
The first and last elements repeat their neighbours' central differences rather than
switching to a one-sided stencil, so the result keeps the input's length and scale.
The dimension to perform differencing over can be specified as `dims`, and the differencing function can be specified as `grad` (defaulting to the euclidean distance, `-`)
"""
centraldiff!(args...; kwargs...) = _diff!(args..., _centraldiff!; kwargs...)

function _diff(x::RegularTimeseries, f!; kwargs...)
    y = deepcopy(x)
    f!(y; kwargs...)
    return y
end
"""
    centraldiff(x::RegularTimeseries; dims=𝑡, grad=-)

Compute the central difference of a regular time series `x`.
The first and last elements repeat their neighbours' central differences rather than
switching to a one-sided stencil, so the result keeps the input's length and scale.
The dimension to perform differencing over can be specified as `dims`, and the differencing function can be specified as `grad` (defaulting to the euclidean distance, `-`)
See [`centraldiff!`](@ref).
"""
centraldiff(args...; kwargs...) = _diff(args..., centraldiff!; kwargs...)

function checkderivdims(dims)
    return if dims isa Tuple || dims isa AbstractVector
        error("Only one dimension can be specified for derivatives.")
    end
end

function _deriv!(x::RegularTimeseries, f!; dims = 𝑡, kwargs...)
    checkderivdims(dims)
    dt = step(x; dims)
    dt isa Dates.Period && throw(
        ArgumentError(
            "in-place derivatives are not supported for a Dates lookup: the result " *
                "carries rate units that the input array cannot store. Use the " *
                "out-of-place form instead."
        )
    )
    f!(x; dims, kwargs...)
    x ./= dt
    return nothing
end

"""
    centralderiv!(x::RegularTimeseries; kwargs...)

Compute the central derivative of a regular time series `x`, in-place.
See [`centraldiff!`](@ref) for available keyword arguments.
"""
centralderiv!(args...; kwargs...) = _deriv!(args..., centraldiff!; kwargs...)

function _deriv(x::RegularTimeseries, f!; dims = 𝑡, kwargs...)
    if step(x; dims) isa Dates.Period
        # A Dates lookup cannot be divided into. Swap in the equivalent elapsed-time
        # axis (which rejects a calendar period), differentiate through the ordinary
        # Unitful path, then restore the original axis.
        q = _stepquantity(step(x; dims))
        lk = parent(lookup(x, dims))
        y = set(x, dims => range(Base.zero(q), step = q, length = size(x, dims)))
        return set(_deriv(y, f!; dims, kwargs...), dims => lk)
    end
    y = deepcopy(x)
    if _unit(step(x; dims)) == NoUnits # Can safely mutate
        f!(y; dims, kwargs...)
    else
        y = ustripall(y)
        f!(y; dims, kwargs...)
        newu = _unit(eltype(x)) / _unit(step(x; dims))
        y = set(x, y .* newu)
    end
    return y
end
"""
    centralderiv(x::AbstractTimeseries)

Compute the central derivative of a time series `x`.
See [`centraldiff`](@ref) for available keyword arguments.
Also c.f. [`centralderiv!`](@ref).
"""
centralderiv(args...; kwargs...) = _deriv(args..., centralderiv!; kwargs...)

function _rightdiff!(x; grad = -, dims = nothing) # Dims unused
    x[1:(end - 1)] .= grad(x[2:end], x[1:(end - 1)])
    # x[[1, end]] .= [grad(a, x[1]), grad(x[end], b)]
    x[[end]] .= [copy(x[end - 1])]
    return nothing
end
"""
    rightdiff!(x::RegularTimeseries; dims=𝑡, grad=-)

In-place [`rightdiff`](@ref): the forward difference `x[i+1] - x[i]`, written back into
`x`. The last element repeats its neighbour, so the length is unchanged.
"""
rightdiff!(args...; kwargs...) = _diff!(args..., _rightdiff!; kwargs...)

"""
    rightdiff(x::RegularTimeseries; dims=𝑡, grad=-)

The forward difference `x[i+1] - x[i]` of a regular time series, as a new series of the
same length. The last element repeats its neighbour, since no forward difference exists
there. Differencing runs along `dims`, and `grad` supplies the difference operator
(the default `-` is the Euclidean one; see [`phasegrad`](@ref) for a circular metric).

See [`rightdiff!`](@ref) for the in-place form, [`rightderiv`](@ref) to divide by the
time step, and [`centraldiff`](@ref)/[`leftdiff`](@ref) for the other stencils.
"""
rightdiff(args...; kwargs...) = _diff(args..., rightdiff!; kwargs...)

"""
    rightderiv!(x::RegularTimeseries; dims=𝑡, grad=-)

In-place [`rightderiv`](@ref). See [`rightdiff!`](@ref) for the available keyword
arguments.
"""
rightderiv!(args...; kwargs...) = _deriv!(args..., rightdiff!; kwargs...)

"""
    rightderiv(x::RegularTimeseries; dims=𝑡, grad=-)

The forward derivative of a regular time series: [`rightdiff`](@ref) divided by the
sampling period, so the result carries units of `eltype(x)` per unit time.

See also [`rightderiv!`](@ref), [`centralderiv`](@ref), [`leftderiv`](@ref).
"""
rightderiv(args...; kwargs...) = _deriv(args..., rightderiv!; kwargs...)

function _leftdiff!(x; grad = -, dims = nothing) # Dims unused
    x[2:end] .= grad(x[2:end], x[1:(end - 1)])
    # x[[1, end]] .= [grad(a, x[1]), grad(x[end], b)]
    x[[1]] .= [copy(x[2])]
    return nothing
end
"""
    leftdiff!(x::RegularTimeseries; dims=𝑡, grad=-)

In-place [`leftdiff`](@ref): the backward difference `x[i] - x[i-1]`, written back into
`x`. The first element repeats its neighbour, so the length is unchanged.
"""
leftdiff!(args...; kwargs...) = _diff!(args..., _leftdiff!; kwargs...)

"""
    leftdiff(x::RegularTimeseries; dims=𝑡, grad=-)

The backward difference `x[i] - x[i-1]` of a regular time series, as a new series of the
same length. The first element repeats its neighbour, since no backward difference exists
there. Differencing runs along `dims`, and `grad` supplies the difference operator
(the default `-` is the Euclidean one; see [`phasegrad`](@ref) for a circular metric).

See [`leftdiff!`](@ref) for the in-place form, [`leftderiv`](@ref) to divide by the time
step, and [`centraldiff`](@ref)/[`rightdiff`](@ref) for the other stencils.
"""
leftdiff(args...; kwargs...) = _diff(args..., leftdiff!; kwargs...)

"""
    leftderiv!(x::RegularTimeseries; dims=𝑡, grad=-)

In-place [`leftderiv`](@ref). See [`leftdiff!`](@ref) for the available keyword
arguments.
"""
leftderiv!(args...; kwargs...) = _deriv!(args..., leftdiff!; kwargs...)

"""
    leftderiv(x::RegularTimeseries; dims=𝑡, grad=-)

The backward derivative of a regular time series: [`leftdiff`](@ref) divided by the
sampling period, so the result carries units of `eltype(x)` per unit time.

See also [`leftderiv!`](@ref), [`centralderiv`](@ref), [`rightderiv`](@ref).
"""
leftderiv(args...; kwargs...) = _deriv(args..., leftderiv!; kwargs...)

Base.abs(x::AbstractTimeseries) = Base.abs.(x)
Base.angle(x::AbstractTimeseries) = Base.angle.(x)

# * See https://en.wikipedia.org/wiki/Directional_statistics

"""
    resultant(θ; dims...)

The mean resultant vector of a sample of angles `θ` (in radians), `mean(exp.(im .* θ))`.
Its length and argument summarise the concentration and mean direction of a circular
distribution. `dims` is passed through to `mean`.

See also [`resultantlength`](@ref), [`circularmean`](@ref).
"""
resultant(θ; kwargs...) = mean(exp.(im .* θ); kwargs...)

"""
    resultantlength(θ; dims...)

The length of the mean [`resultant`](@ref) vector of angles `θ`, in `[0, 1]`: near `0` for
uniformly spread angles, near `1` for tightly concentrated angles.
"""
resultantlength(θ; kwargs...) = min.(1, abs.(resultant(θ; kwargs...)))

"""
    circularmean(θ; dims...)

The circular mean of angles `θ` (radians): the argument of the mean [`resultant`](@ref)
vector, in `(-π, π]`.
"""
circularmean(θ; kwargs...) = angle.(resultant(θ; kwargs...))

"""
    circularvar(θ; dims...)

The circular variance of angles `θ`, `1 - resultantlength(θ)`, in `[0, 1]`.
"""
circularvar(θ; kwargs...) = 1 .- resultantlength(θ; kwargs...)

"""
    circularstd(θ; dims...)

The circular standard deviation of angles `θ`, `sqrt(-2 log(resultantlength(θ)))`.
"""
circularstd(θ; kwargs...) = sqrt.(-2 * log.(resultantlength(θ; kwargs...)))

"""
    addrefdim(X::AbstractDimArray, dim::DimensionalData.Dimension)

Return `X` with `dim` appended to its reference dimensions (`refdims`).
"""
function addrefdim(X::AbstractDimArray, dim::DimensionalData.Dimension)
    return rebuild(
        X; dims = dims(X),
        metadata = DimensionalData.metadata(X),
        name = DimensionalData.name(X),
        refdims = (DimensionalData.refdims(X)..., dim)
    )
end

"""
    addmetadata(X::AbstractDimArray; kwargs...)

Return `X` with the keyword arguments merged into its metadata. Existing entries are kept;
any key that collides with a keyword argument is overwritten (with a warning).
"""
function addmetadata(X::AbstractDimArray; kwargs...)
    md0 = DimensionalData.metadata(X)
    prs = md0 isa DimensionalData.NoMetadata ? Pair{Symbol, Any}[] :
        md0 isa DimensionalData.Metadata ? collect(pairs(md0.val)) :
        collect(pairs(md0))
    dup = intersect(collect(keys(kwargs)), first.(prs))
    isempty(dup) || @warn "Metadata already contains keys, overwriting: $(dup)"
    md = DimensionalData.Metadata(prs..., kwargs...)
    return rebuild(
        X; dims = dims(X),
        metadata = md,
        name = DimensionalData.name(X),
        refdims = DimensionalData.refdims(X)
    )
end

"""
    align(x::AbstractDimArray, ts, dt; dims = 1)

Align a `DimArray` `x` to each of a set of dimension values `ts`, selecting a window given by `dt` centered at each element of `ts`.
`dt` can be a two-element vector/tuple, or an interval.
The `dims` argument specifies the dimension along which the alignment is performed.
Each element of the resulting `DimArray` is an aligned portion of the original `x`.
"""
function align(
        x::DimensionalData.AbstractDimArray, ts,
        dt::Union{<:Tuple, <:AbstractVector}; dims = 1, zero = true
    )
    (dims isa Tuple || dims isa AbstractVector) &&
        throw(ArgumentError("align: only one dimension can be specified, got $dims"))
    dims = DimensionalData.dims(x, dims) # accepts an index, a type, or a dimension
    ints = [Interval((t .+ dt)...) for t in ts]
    x = Timeseries([view(x, rebuild(dims, i)) for i in ints], ts)
    if zero
        x = set(
            x, map(enumerate(x)) do (i, _x)
                set(_x, dims => lookup(_x, dims) .- ts[i])
            end
        )
    end
    return x
end
align(x, ts, dt::Interval; kwargs...) = align(x, ts, extrema(dt); kwargs...)

"""
    stitch(x, args...)

Stitch multiple time series together by concatenating along the time dimension generating new contiguous time indices. An instant (`DateTime`, `Date`) time index continues the calendar from the first series' own start, so the result is still a calendar series; a numeric or elapsed (`Period`) index restarts one step from the origin. The time series must be of the same type (`UnivariateRegular`, `MultivariateRegular`, or `AbstractArray`), and the sampling period and dimensions of the data arrays must match. If the arguments are `MultivariateRegular, they must have the same dimensions (except for the time dimension).

# Arguments
- `X`: The first time series.
- `args...`: Additional time series.

# Returns
- A new time series containing the concatenated data.
"""
# The first index of a stitched series. A numeric or elapsed axis restarts one step in,
# as it always has; an instant axis continues the original calendar rather than dropping
# it, which would silently cost `groupby`, `months` and the instant selectors.
_stitchstart(t, dt) = dt
_stitchstart(t::AbstractVector{<:Dates.TimeType}, dt) = first(t)

function stitch(x::UnivariateRegular, y::UnivariateRegular)
    dt = samplingperiod(x)
    @assert dt == samplingperiod(y)
    z = vcat(x.data, y.data)
    return z = Timeseries(
        z,
        range(_stitchstart(times(x), dt), step = dt, length = size(z, 1))
    )
end
stitch(x::AbstractArray, y::AbstractArray) = vcat(x, y)
function stitch(x::MultivariateRegular, y::MultivariateRegular)
    dt = samplingperiod(x)
    @assert dt == samplingperiod(y)
    @assert all(dims(x)[2:end] .== dims(y)[2:end])
    z = vcat(x.data, y.data)
    return z = Timeseries(
        z,
        range(_stitchstart(times(x), dt), step = dt, length = size(z, 1)),
        dims(x)[2:end]...
    )
end
stitch(X, Y, args...) = reduce(stitch, (X, Y, args...))

"""
    coarsegrain(X::AbstractArray; dims = nothing, newdim=ndims(X)+1)
Coarse-grain an array by taking every second element over the given dimensions `dims` and concatenating them in the dimension `newdim`. `dims` are coarse-grained in sequence, from last to first. If `dims` is not specified, we iterate over all dimensions that are not `newdim`. If the array has an odd number of slices in any `dims`, the last slice is discarded.
This is more flexibile than the conventional, mean-based definition of coarse graining: it can be used to generate coarse-grained distributions from an array. To recover this conventional mean-based coarse-graining:
```julia
    C = coarsegrain(X)
    mean(C, dims=ndims(C))
```
"""
function coarsegrain(X::AbstractArray; dims = nothing, newdim = ndims(X) + 1)
    if isnothing(dims)
        dims = collect(1:ndims(X))
        dims = setdiff(dims, newdim)
    end
    dims = collect(Tuple(dims))
    if newdim ∈ dims
        error("`dims` cannot contain `newdim`")
    end
    all(size(X)[dims] .> 1) ||
        error("Cannot coarse-grain a dimension with only one element")
    while !isempty(dims)
        dim = pop!(dims)
        𝒳 = eachslice(X; dims = dim)
        N = floor(Int, length(𝒳) / 2)
        X = cat(
            stack(𝒳[1:2:(N * 2)], dims = dim), stack(𝒳[2:2:(N * 2)], dims = dim),
            dims = newdim
        )
    end
    return X
end

function coarsegrain(
        X::AbstractDimArray; dims = nothing,
        newdim = ndims(X) + 1
    )
    if isnothing(dims)
        dims = DimensionalData.dims(X)
    end
    _dims = [dimnum(X, dims)...]
    dims = DimensionalData.dims.([X], _dims)
    if hasdim(X, newdim)
        _newdim = dimnum(X, newdim)
        newdim = DimensionalData.dims(X, _newdim)
    else
        _newdim = ndims(X) + 1
    end
    while !isempty(_dims)
        _dim = pop!(_dims)
        _X = coarsegrain(X.data; dims = _dim, newdim = _newdim)
        N = floor(Int, size(X, _dim) / 2)
        if hasdim(X, newdim)
            newdim = rebuild(
                DimensionalData.dims(X, newdim),
                vcat(
                    DimensionalData.dims(X, _newdim).val,
                    DimensionalData.dims(X, _newdim).val
                )
            )
            newdims = collect(Any, DimensionalData.dims(X))
            newdims[_dim] = rebuild(
                newdims[_dim],
                _onoffsets(
                    o -> (o[1:2:(N * 2)] .+ o[2:2:(N * 2)]) ./ 2,
                    parent(newdims[_dim])
                )
            )
            newdims[_newdim] = newdim
        else
            newdims = collect(Any, DimensionalData.dims(X))
            newdims[_dim] = rebuild(
                newdims[_dim],
                _onoffsets(
                    o -> (o[1:2:(N * 2)] .+ o[2:2:(N * 2)]) ./ 2,
                    parent(newdims[_dim])
                )
            )
            newdims = [newdims..., DimensionalData.AnonDim(1:size(_X, _newdim))]
            newdim = newdims[newdim]
        end
        X = ToolsArray(
            _X, Tuple(newdims); refdims = refdims(X), name = name(X),
            metadata = metadata(X)
        )
    end

    return X
end

"""
    Dropdims(f)

Wrap a reducing function `f` so that the reduced dimensions are dropped from the result:
`Dropdims(f)(args...; dims, kwargs...)` is `dropdims(f(args...; dims, kwargs...); dims)`.

## Examples
```julia
julia> Dropdims(sum)(x; dims = 1)   # sum over dimension 1 and drop it
```
"""
struct Dropdims <: Function
    f::Any
end
function (d::Dropdims)(args...; dims, kwargs...)
    return Base.dropdims(d.f(args...; dims, kwargs...); dims)
end

end
