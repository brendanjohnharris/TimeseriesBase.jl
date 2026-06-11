module Utils

using IntervalSets
using DimensionalData
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.UnitfulTools

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
[:($(S)(D::Dimension) = $(S)(D.val.data)) for S in Selectors] .|> eval

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
    step(x::RegularTimeseries)

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
"""
samplingrate(x::RegularTimeseries; kwargs...) = 1 / step(x; kwargs...)

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
    isempty(y) && return Timeseries(y, times(x)[1:0])
    t = _buffer(times(x), args...; kwargs...) .|> mean
    # For a regular time series, the buffer centres are regular
    ts = range(first(t), last(t), length(y))
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
    δt = τ * p * step(x)
    delays = (-(δt * (n - 1))):δt:0
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

# Bare-bones lookup → (regular_range, orig_values, max_dev, worst_idx).
# `orig` is the untouched input collected to a vector so callers that want to
# preserve the raw lookup (e.g. `zero=true`) have it.
function _fit_regular_grid(ts::AbstractVector; atol = nothing, sigdigits = nothing)
    n = length(ts)
    n < 2 && throw(ArgumentError("regularize: need at least 2 lookup points, got $n"))
    u = unit(eltype(ts))
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
    fit = range(start = t0bare, step = stpbare, length = n)
    devs = abs.(tsbare .- fit)
    maxdev, worst = findmax(devs)

    # Rounding precision. The legacy code rounded to a fixed decimal-digit
    # count, which silently couples to the magnitude of the step. We instead
    # round start and step to the decimal place implied by `q` (see
    # `_round_to`), where `q` is one tenth of `atol` — so rounding error is
    # always < atol/10, well below the acceptance threshold, while the
    # precision still scales with the data. `sigdigits` is a manual override.
    atolbare = atol === nothing ? nothing : ustripall(atol)
    t0_r, stp_r = if sigdigits !== nothing
        round(t0bare; sigdigits), round(stpbare; sigdigits)
    elseif atolbare !== nothing && atolbare > 0
        q = atolbare / 10
        _round_to(t0bare, q), _round_to(stpbare, q)
    else
        # No tolerance given: round at `default_atol/10`, where
        # default_atol = step·1e-6 (see `_default_atol`). This keeps the
        # rounding error well below the regularity threshold and lets
        # near-integer / near-rational lookups snap back to clean values.
        q = abs(stpbare) * 1.0e-7
        _round_to(t0bare, q), _round_to(stpbare, q)
    end

    grid_bare = range(start = t0_r, step = stp_r, length = n)
    grid = u == NoUnits ? grid_bare : grid_bare .* u
    maxdev_u = u == NoUnits ? maxdev : maxdev * u
    return grid, collect(ts), maxdev_u, worst
end

# Default tolerance: 1e-6 of the step magnitude. Tight enough to catch real
# irregularity, loose enough to absorb the float jitter that motivates calling
# this in the first place. Caller should pass the LSQ-fitted step (with units)
# so we don't re-fit.
_default_atol(step_with_units) = 1.0e-6 * abs(step_with_units)

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
  best-fit regular grid. Carries units if the lookup does. If `nothing`, a
  permissive `~1e3 * eps` floor is used.
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
    regularize(X::AbstractDimArray; dims=𝑡, atol=nothing, sigdigits=nothing,
               zero=false, strict=true)

Replace the lookup of `X` along each of `dims` with a regular range that best
fits the existing values, repairing accumulated float jitter. If a lookup
deviates from regular by more than `atol` an `ArgumentError` is thrown
(`strict=false` downgrades this to a warning).

If `zero=true`, the new lookup starts at zero and the genuine original
lookup is stored in `metadata(X)` under the dimension name.

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
        new_grid = zero ? range(
                start = Base.zero(first(grid)), step = step(grid),
                length = length(grid)
            ) : grid
        X = set(X, dim => parent(new_grid))
        if zero
            X = rebuild(
                X;
                metadata = Dict(
                    Symbol(DimensionalData.name(d)) => orig,
                    pairs(metadata(X))...,
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
        pad = minimum(
            median(abs.(diff(collect(parent(lookup(x, dim)))))) for x in Xs
        ) / 2
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
        grid, _ = _fit_regular_grid(mean_lookup; atol, sigdigits)

        # Verify every input is within tolerance of the common grid and
        # report the worst offender by array index.
        dim_label = DimensionalData.name(DimensionalData.dims(Xs[1], dim))
        tol = atol === nothing ? _default_atol(step(grid)) : atol
        for (i, xl) in enumerate(lookups)
            devs = abs.(xl .- grid)
            maxdev, worst = findmax(devs)
            if maxdev > tol
                msg = "regularize: array $i lookup along $dim_label deviates " *
                    "from common grid by $maxdev > atol=$tol " *
                    "(worst at index $worst, value $(xl[worst]))"
                strict ? throw(ArgumentError(msg)) : @warn msg
            end
        end

        # Apply the (possibly zero-shifted) grid and stash the *genuine*
        # original lookup (captured before `set`).
        new_grid = zero ? range(
                start = Base.zero(first(grid)), step = step(grid),
                length = length(grid)
            ) : grid
        Xs = map(zip(Xs, lookups)) do (x, origlk)
            x = set(x, dim => parent(new_grid))
            zero ?
                rebuild(
                    x;
                    metadata = Dict(Symbol(dim_label) => origlk, pairs(metadata(x))...)
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
figures. The array forms operate along `dims`; the vararg form additionally aligns several
arrays onto a shared grid. With `zero=true` the lookup starts at zero and the original is
stored in metadata.

!!! note
    Superseded by [`regularize`](@ref), which fits the grid by least squares, checks
    regularity by maximum deviation, and throws (rather than warns) on failure by default.
    Prefer `regularize` for new code.
"""
function rectify(ts::DimensionalData.Dimension; tol=4, zero=false, extend=false,
    atol=nothing)
    u = unit(eltype(ts))
    ts = collect(ts)
    origts = ts
    stp = ts |> diff |> mean
    err = ts |> diff |> std
    tol = Int(tol - round(log10(stp |> ustripall)))

    if isnothing(atol) && ustripall(err) > exp10(-tol - 1)
        @warn "Step $stp is not approximately constant (err=$err, tol=$(exp10(-tol-1))), skipping rectification"
    else
        if !isnothing(atol)
            tol = atol
        end
        stp = u == NoUnits ? round(stp; digits=tol) : round(u, stp; digits=tol)
        t0, t1 = u == NoUnits ? round.(extrema(ts); digits=tol) :
                 round.(u, extrema(ts); digits=tol)
        if zero
            origts = t0:stp:(t1+(10000*stp))
            t1 = t1 - t0
            t0 = 0
        end
        if extend
            ts = t0:stp:(t1+(10000*stp))
        else
            ts = range(start=t0, step=stp, length=length(ts))
        end
    end
    return parent(ts), origts
end

function rectify(X::AbstractDimArray; dims, tol=4, zero=false, kwargs...) # tol gives significant figures for rounding
    for dim in _dimlist(dims)
        ts, origts = rectify(DimensionalData.dims(X, dim); tol, zero, extend=true,
            kwargs...)
        ts = ts[1:size(X, dim)] # Should be ok?
        @assert length(ts) == size(X, dim)
        X = set(X, dim => ts)
        @assert lookup(X, dim) == ts
        if zero
            X = rebuild(X; metadata=(Symbol(dim) => origts, pairs(metadata(X))...))
        end
    end
    return X
end

function rectify(X::Vararg{AbstractDimArray}; dims=𝑡, tol=4, zero=false,
    kwargs...)

    # Process each dimension
    for dim in _dimlist(dims)
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
        u = unit(eltype(all_dims[1]))
        tol_val = u == NoUnits ? exp10(-tol) : exp10(-tol) * u
        common_range = (mint - tol_val) .. (maxt + tol_val)
        X = [x[dim(common_range)] for x in X]

        # Find minimum common length
        min_length = minimum([size(x, dim) for x in X])
        X = [selectdim(x, dimnum(x, dim), 1:min_length) for x in X]

        # Compute mean dimension values for rectification
        mean_dim_vals = mean([collect(DimensionalData.dims(x, dim)) for x in X])

        # Rectify using the mean dimension values
        ts, origts = rectify(rebuild(DimensionalData.dims(X[1], dim), mean_dim_vals);
            tol=tol, zero=zero, extend=true, kwargs...)

        # Trim to actual size
        ts = ts[1:min_length]

        # Verify rectification is within tolerance
        for (i, x) in enumerate(X)
            x_dims = collect(DimensionalData.dims(x, dim))
            max_diff = maximum(abs.(ts .- x_dims))
            # Base tolerance on rectification precision, not step variability
            u = unit(eltype(x_dims))
            expected_tol = u == NoUnits ? exp10(-tol) : exp10(-tol) * u

            if max_diff > expected_tol
                @warn "Array $i: dimension $dim differs from rectified values by up to $max_diff (tolerance: $expected_tol)"
            end
        end

        # Apply rectified dimension to all arrays
        X = [set(x, dim => ts) for x in X]

        # Add original dimension values to metadata if zero=true
        if zero
            X = [rebuild(x;
                metadata=(Symbol(dim) => origts[1:min_length],
                    pairs(metadata(x))...)) for x in X]
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
rectifytime(X::Vararg{AbstractTimeseries}; kwargs...) = rectify(X...; dims=𝑡, kwargs...)

"""
    matchdim(X::AbstractVector{<:AbstractDimArray}; dims=1, tol=4, zero=false)

Align a collection of dimensional arrays onto a common rectified grid along `dims`, so that
every element of `X` shares an identical lookup.

!!! note
    Superseded by [`regularize`](@ref), which fits the grid by least squares and applies a
    stricter regularity check; prefer `regularize` for new code.
"""
function matchdim(X::AbstractVector{<:AbstractDimArray}; dims=1, tol=4, zero=false,
    kwargs...)
    # Generate some common time indices as close as possible to the rectified times of each element of the input vector. At most this will change each time index by a maximum of 1 sampling period. We could do better--maximum of a half-- but leave that for now.
    u = lookup(X |> first, dims) |> eltype |> unit
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
    ts, origts = rectify(rebuild(DimensionalData.dims(X[1], dims), ts); tol, zero,
        kwargs...)
    if any([any(ts .- lookup(x, dims) .> std(ts) / exp10(-tol)) for x in X])
        @error "Cannot find common dimension indices within tolerance"
    end
    X = [set(x, rebuild(DimensionalData.dims(x, dims), ts)) for x in X]
    return X
end

"""
    phasegrad(x, y)

The signed circular difference between angles `x` and `y` (in radians), wrapped to
`(-π, π]`: the shortest signed rotation from `y` to `x`. For example
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
    return f!(parent(eachslice(x; dims)); kwargs...)
end

"""
    centraldiff!(x::RegularTimeseries; dims=𝑡, grad=-)

Compute the central difference of a regular time series `x`, in-place.
The first and last elements are set to the forward and backward difference, respectively.
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
The first and last elements are set to the forward and backward difference, respectively.
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
    f!(x; dims, kwargs...)
    x ./= step(x; dims)
    return nothing
end

"""
    centralderiv!(x::RegularTimeseries; kwargs...)

Compute the central derivative of a regular time series `x`, in-place.
See [`centraldiff!`](@ref) for available keyword arguments.
"""
centralderiv!(args...; kwargs...) = _deriv!(args..., centraldiff!; kwargs...)

function _deriv(x::RegularTimeseries, f!; dims = 𝑡, kwargs...)
    y = deepcopy(x)
    if unit(step(x; dims)) == NoUnits # Can safely mutate
        f!(y; dims, kwargs...)
    else
        y = ustripall(y)
        f!(y; dims, kwargs...)
        newu = unit(eltype(x)) / unit(step(x; dims))
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
rightdiff!(args...; kwargs...) = _diff!(args..., _rightdiff!; kwargs...)
rightdiff(args...; kwargs...) = _diff(args..., rightdiff!; kwargs...)
rightderiv!(args...; kwargs...) = _deriv!(args..., rightdiff!; kwargs...)
rightderiv(args...; kwargs...) = _deriv(args..., rightderiv!; kwargs...)

function _leftdiff!(x; grad = -, dims = nothing) # Dims unused
    x[2:end] .= grad(x[2:end], x[1:(end - 1)])
    # x[[1, end]] .= [grad(a, x[1]), grad(x[end], b)]
    x[[1]] .= [copy(x[2])]
    return nothing
end
leftdiff!(args...; kwargs...) = _diff!(args..., _leftdiff!; kwargs...)
leftdiff(args...; kwargs...) = _diff(args..., leftdiff!; kwargs...)
leftderiv!(args...; kwargs...) = _deriv!(args..., leftdiff!; kwargs...)
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
resultantlength(θ; kwargs...) = abs.(resultant(θ; kwargs...))

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
circularvar(θ; kwargs...) = 1 - resultantlength(θ; kwargs...)

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
    @assert length(dims) == 1
    dims isa Integer &&
        (dims = DimensionalData.dims(x, dims))
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

Stitch multiple time series together by concatenating along the time dimension generating new contiguous time indices. The time series must be of the same type (`UnivariateRegular`, `MultivariateRegular`, or `AbstractArray`), and the sampling period and dimensions of the data arrays must match. If the arguments are `MultivariateRegular, they must have the same dimensions (except for the time dimension).

# Arguments
- `X`: The first time series.
- `args...`: Additional time series.

# Returns
- A new time series containing the concatenated data.
"""
function stitch(x::UnivariateRegular, y::UnivariateRegular)
    dt = samplingperiod(x)
    @assert dt == samplingperiod(y)
    z = vcat(x.data, y.data)
    return z = Timeseries(z, dt:dt:(dt * size(z, 1)))
end
stitch(x::AbstractArray, y::AbstractArray) = vcat(x, y)
function stitch(x::MultivariateRegular, y::MultivariateRegular)
    dt = samplingperiod(x)
    @assert dt == samplingperiod(y)
    @assert all(dims(x)[2:end] .== dims(y)[2:end])
    z = vcat(x.data, y.data)
    return z = Timeseries(z, dt:dt:(dt * size(z, 1)), dims(x)[2:end]...)
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
                (
                    parent(newdims[_dim][1:2:(N * 2)]) +
                        parent(newdims[_dim][2:2:(N * 2)])
                ) / 2
            )
            newdims[_newdim] = newdim
        else
            newdims = collect(Any, DimensionalData.dims(X))
            newdims[_dim] = rebuild(
                newdims[_dim],
                (
                    parent(newdims[_dim][1:2:(N * 2)]) +
                        parent(newdims[_dim][2:2:(N * 2)])
                ) / 2
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
function (d::Dropdims)(args...; dims = :, kwargs...)
    return Base.dropdims(d.f(args...; dims, kwargs...); dims)
end

end
