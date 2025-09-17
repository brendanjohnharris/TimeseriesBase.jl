module Utils

using IntervalSets
using DimensionalData
using TimeseriesBase.ToolsArrays
using TimeseriesBase.TimeSeries
using TimeseriesBase.UnitfulTools

import DimensionalData.Dimensions.LookupArrays: At, Near
import DimensionalData.Dimensions.Dimension
import DimensionalData: print_array, _print_array_ctx, _print_indices_vec
import DimensionalData: At, Between, Touches, Near, Where, Contains
using Unitful
using Statistics

export times, step, samplingrate, samplingperiod, duration, coarsegrain, stitch,
       buffer, window, delayembed, rectifytime, rectify, matchdim, interlace,
       centraldiff!, centraldiff, centralderiv!, centralderiv,
       rightdiff!, rightdiff, rightderiv!, rightderiv,
       leftdiff!, leftdiff, leftderiv!, leftderiv,
       abs, angle, resultant, resultantlength,
       circularmean, circularvar, circularstd,
       phasegrad,
       addrefdim, addmetadata, align

# import LinearAlgebra.mul!
# function mul!(a::AbstractVector, b::AbstractTimeseries, args...; kwargs...)
#     mul!(a, b.data, args...; kwargs...)
# end

Selectors = [:At, :Between, :Touches, :Near, :Where, :Contains]
# Allow dims to be passed directly to selectors
[:($(S)(D::Dimension) = $(S)(D.val.data)) for S in Selectors] .|> eval

description(x) = "$(size(x)) $(typeof(x).name.name)"
function print_array(io::IO, mime, A::AbstractDimArray{T, 0}) where {T <: AbstractArray}
    print(_print_array_ctx(io, T), "\n", description.(A[]))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 1}) where {T <: AbstractArray}
    Base.print_matrix(_print_array_ctx(io, T), description.(A))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 2}) where {T <: AbstractArray}
    Base.print_matrix(_print_array_ctx(io, T), description.(A))
end
function print_array(io::IO, mime, A::AbstractToolsArray{T, 3}) where {T <: AbstractArray}
    i3 = firstindex(A, 3)
    frame = view(A, :, :, i3)
    _print_indices_vec(io, i3)
    Base.print_matrix(_print_array_ctx(io, T), description.(frame))
end
function print_array(io::IO, mime,
                     A::AbstractToolsArray{T, N}) where {T <: AbstractArray, N}
    o = ntuple(x -> firstindex(A, x + 2), N - 2)
    frame = view(A, :, :, o...)

    _print_indices_vec(io, o...)
    Base.print_matrix(_print_array_ctx(io, T), description.(frame))
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

function interlace(x::AbstractTimeseries, y::AbstractTimeseries)
    ts = vcat(times(x), times(y))
    idxs = sortperm(ts)
    ts = ts[idxs]
    data = vcat(x.data, y.data)
    data = data[idxs]
    return Timeseries(data, ts)
end

function _buffer(x, n::Integer, p::Integer = 0; discard::Bool = true)
    y = [@views x[i:min(i + n - 1, end)] for i in 1:(n - p):length(x)]
    while discard && length(y[end]) < n
        pop!(y)
    end
    y
end
function _buffer(x::AbstractMatrix, n::Integer, p::Integer = 0; discard::Bool = true)
    y = [@views x[i:min(i + n - 1, end), :] for i in 1:(n - p):size(x, 1)]
    while discard && size(y[end], 1) < n
        pop!(y)
    end
    y
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
    t = _buffer(times(x), args...; kwargs...) .|> mean
    # For a regular time series, the buffer centres are regular
    ts = range(first(t), last(t), length(y))
    y = Timeseries(y, ts)
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
    y = map(y) do _y
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
    y = stack(y, dims = 1) # dims=1 so time is on first dimension
end

function rectify(ts::DimensionalData.Dimension; tol = 4, zero = false, extend = false,
                 atol = nothing)
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
        stp = u == NoUnits ? round(stp; digits = tol) : round(u, stp; digits = tol)
        t0, t1 = u == NoUnits ? round.(extrema(ts); digits = tol) :
                 round.(u, extrema(ts); digits = tol)
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
    return ts, origts
end

function rectify(X::AbstractDimArray; dims, tol = 4, zero = false, kwargs...) # tol gives significant figures for rounding
    if !(dims isa Tuple || dims isa AbstractVector)
        dims = [dims]
    end
    for dim in dims
        ts, origts = rectify(DimensionalData.dims(X, dim); tol, zero, extend = true,
                             kwargs...)
        ts = ts[1:size(X, dim)] # Should be ok?
        @assert length(ts) == size(X, dim)
        X = set(X, dim => ts)
        if zero
            X = rebuild(X; metadata = (Symbol(dim) => origts, pairs(metadata(X))...))
        end
    end
    return X
end

function rectify(X::Vararg{AbstractDimArray}; dims = 𝑡, tol = 4, zero = false,
                 kwargs...)

    # Ensure dims is iterable
    if !(dims isa Tuple || dims isa AbstractVector)
        dims = [dims]
    end

    # Process each dimension
    for dim in dims
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
                             tol = tol, zero = zero, extend = true, kwargs...)

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
                         metadata = (Symbol(dim) => origts[1:min_length],
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
rectifytime(X::Vararg{AbstractTimeseries}; kwargs...) = rectify(X...; dims = 𝑡, kwargs...)

function matchdim(X::AbstractVector{<:AbstractDimArray}; dims = 1, tol = 4, zero = false,
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

phasegrad(x::Real, y::Real) = mod(x - y + π, 2π) - π # +pi - pi because we want the difference mapped from -pi to +pi, so we can represent negative changes.
phasegrad(x, y) = phasegrad.(x, y)
phasegrad(x::Complex, y::Complex) = phasegrad(angle(x), angle(y))

function _centraldiff!(x; grad = -, dims = nothing) # Dims unused
    # a = x[2] # Save here, otherwise they get mutated before we use them
    # b = x[end - 1]
    if grad == -
        x[2:(end - 1)] .= grad(x[3:end], x[1:(end - 2)]) / 2
    else # For a non-euclidean metric, we need to calculate both sides individually
        x[2:(end - 1)] .= (grad(x[3:end], x[2:(end - 1)]) +
                           grad(x[2:(end - 1)], x[1:(end - 2)])) / 2
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
    f!(parent(eachslice(x; dims)); kwargs...)
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
    if dims isa Tuple || dims isa AbstractVector
        error("Only one dimension can be specified for derivatives.")
    end
end

function _deriv!(x::RegularTimeseries, f!; dims = 𝑡, kwargs...)
    checkderivdims(dims)
    f!(x; dims, kwargs...)
    x ./= step(x; dims)
    nothing
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
resultant(θ; kwargs...) = mean(exp.(im .* θ); kwargs...)
resultantlength(θ; kwargs...) = abs.(resultant(θ; kwargs...))
circularmean(θ; kwargs...) = angle.(resultant(θ; kwargs...))
circularvar(θ; kwargs...) = 1 - resultantlength(θ; kwargs...)
circularstd(θ; kwargs...) = sqrt.(-2 * log.(resultantlength(θ; kwargs...)))

## Add refdims to a DimArray
function addrefdim(X::AbstractDimArray, dim::DimensionalData.Dimension)
    rebuild(X; dims = dims(X),
            metadata = DimensionalData.metadata(X),
            name = DimensionalData.name(X),
            refdims = (DimensionalData.refdims(X)..., dim))
end

function addmetadata(X::AbstractDimArray; kwargs...)
    p = DimensionalData.metadata(X)
    p = p isa DimensionalData.Metadata ? pairs(p.val) : pairs(p)
    if any(keys(kwargs) .∈ [keys(p)])
        @warn "Metadata already contains one of the keys, overwriting $(collect(pairs(kwargs))[keys(kwargs) .∈ [keys(p)]])"
    end
    md = DimensionalData.Metadata(p..., kwargs...)
    rebuild(X; dims = dims(X),
            metadata = md,
            name = DimensionalData.name(X),
            refdims = DimensionalData.refdims(X))
end

"""
    align(x::AbstractDimArray, ts, dt; dims = 1)

Align a `DimArray` `x` to each of a set of dimension values `ts`, selecting a window given by `dt` centered at each element of `ts`.
`dt` can be a two-element vector/tuple, or an interval.
The `dims` argument specifies the dimension along which the alignment is performed.
Each element of the resulting `DimArray` is an aligned portion of the original `x`.
"""
function align(x::DimensionalData.AbstractDimArray, ts,
               dt::Union{<:Tuple, <:AbstractVector}; dims = 1, zero = true)
    @assert length(dims) == 1
    dims isa Integer &&
        (dims = DimensionalData.dims(x, dims))
    ints = [Interval((t .+ dt)...) for t in ts]
    x = Timeseries([view(x, rebuild(dims, i)) for i in ints], ts)
    if zero
        x = set(x, map(enumerate(x)) do (i, _x)
                    set(_x, dims => lookup(_x, dims) .- ts[i])
                end)
    end
    return x
end
align(x, ts, dt::Interval; kwargs...) = align(x, ts, extrema(dt); kwargs...)

function upsample(d::DimensionalData.Dimension{<:RegularIndex}, factor::Number)
    rebuild(d, range(start = minimum(d), stop = maximum(d), step = step(d) / factor))
end
function upsample(d::DimensionalData.Dimension, factor)
    rebuild(d,
            range(start = minimum(d), stop = maximum(d),
                  step = mean(diff(lookup(d))) / factor))
end

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
    z = Timeseries(z, dt:dt:(dt * size(z, 1)))
end
stitch(x::AbstractArray, y::AbstractArray) = vcat(x, y)
function stitch(x::MultivariateRegular, y::MultivariateRegular)
    dt = samplingperiod(x)
    @assert dt == samplingperiod(y)
    @assert all(dims(x)[2:end] .== dims(y)[2:end])
    z = vcat(x.data, y.data)
    z = Timeseries(z, dt:dt:(dt * size(z, 1)), dims(x)[2:end]...)
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
        X = cat(stack(𝒳[1:2:(N * 2)], dims = dim), stack(𝒳[2:2:(N * 2)], dims = dim),
                dims = newdim)
    end
    return X
end

function coarsegrain(X::AbstractDimArray; dims = nothing,
                     newdim = ndims(X) + 1)
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
            newdim = rebuild(DimensionalData.dims(X, newdim),
                             vcat(DimensionalData.dims(X, _newdim).val,
                                  DimensionalData.dims(X, _newdim).val))
            newdims = collect(Any, DimensionalData.dims(X))
            newdims[_dim] = rebuild(newdims[_dim],
                                    (parent(newdims[_dim][1:2:(N * 2)]) +
                                     parent(newdims[_dim][2:2:(N * 2)])) / 2)
            newdims[_newdim] = newdim
        else
            newdims = collect(Any, DimensionalData.dims(X))
            newdims[_dim] = rebuild(newdims[_dim],
                                    (parent(newdims[_dim][1:2:(N * 2)]) +
                                     parent(newdims[_dim][2:2:(N * 2)])) / 2)
            newdims = [newdims..., DimensionalData.AnonDim(1:size(_X, _newdim))]
            newdim = newdims[newdim]
        end
        X = ToolsArray(_X, Tuple(newdims); refdims = refdims(X), name = name(X),
                       metadata = metadata(X))
    end

    return X
end
end
