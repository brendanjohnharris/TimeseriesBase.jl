module Operators
using ..Utils
using ..ToolsArrays
using DimensionalData

export ℬ, ℬ!, ℒ!, ℒ, 𝒯

# ? Some basic time-series operators

"""
    ℬ!(x, [n=1])

In-place [`ℬ`](@ref): circularly shift the elements of `x` forward by `n` positions
(default `1`).
"""
ℬ!(x) = circshift!(x, 1)
ℬ!(x, n) = circshift!(x, n)

"""
    ℬ(x, [n=1])

The backshift operator: a copy of `x` with its elements circularly shifted forward by `n`
positions (default `1`), so that `ℬ(x)[i] == x[i-1]`. See [`ℬ!`](@ref) for the in-place
form, and [`ℒ`](@ref), [`𝒯`](@ref).
"""
ℬ(x, args...) = (x = deepcopy(x);
                 ℬ!(x, args...);
                 x)

"""
    ℒ!(x, [n=1])

In-place [`ℒ`](@ref): circularly shift the elements of `x` backward by `n` positions
(default `1`).
"""
ℒ!(x) = circshift!(x, -1)
ℒ!(x, n) = circshift!(x, -n)

"""
    ℒ(x, [n=1])

The lead operator: a copy of `x` with its elements circularly shifted backward by `n`
positions (default `1`), so that `ℒ(x)[i] == x[i+1]`; the inverse of [`ℬ`](@ref). See
[`ℒ!`](@ref) for the in-place form.
"""
ℒ(x, args...) = (y = deepcopy(x); ℒ!(y, args...); y)

"""
    𝒯(x, t)
    𝒯(t)

The time-shift operator: translate the time index of `x` by `t`, returning a time series
whose `times` are `times(x) .+ t` (the data is unchanged). The one-argument form `𝒯(t)`
returns a function that applies this shift.

See also [`ℬ`](@ref), [`ℒ`](@ref).
"""
𝒯(x, t) = set(x, 𝑡(times(x) .+ t))
𝒯(t) = Base.Fix2(𝒯, t)

end
