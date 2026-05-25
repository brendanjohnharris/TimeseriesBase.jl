@testitem "Interlace" tags=[:fast] begin
    x = Timeseries(randn(11), 0:0.1:1)
    y = Timeseries(randn(10), 0.05:0.1:1)
    z = @test_nowarn interlace(x, y)
    @test all(collect(times(z)) .== 0.0:0.05:1.0)
end

@testitem "Cat and stack" tags=[:fast] begin
    x = Timeseries(randn(100, 100), 0.1:0.1:10, Var(1:100))
    y = cat(x, x; dims = 𝑓(1:2))
    @test dims(y, 3) == 𝑓(1:2)

    z = ToolsArray([x, x], 𝑓(1:2))
    z = stack(z)
    @test z isa ToolsArray
    @test y == z

    z = ToolsArray([x, x], 𝑓(1:2))
    y = stack(z; dims = 1)
    @test y isa ToolsArray
    @test dims(y, 1) == 𝑓(1:2)
end

@testitem "Buffer" tags=[:fast] begin
    N = 10
    x = Timeseries(randn(100), 0.1:0.1:10)
    y = @test_nowarn buffer(x, 10)
    @test length(y) == N
    @test y[1] == x[1:(length(x) ÷ N)]
    @test cat(y..., dims = 𝑡) == x[1:((length(x) ÷ N) * N)]

    y = @test_nowarn buffer(x, 10, 0; discard = false)
    @test cat(y..., dims = 𝑡) == x

    y = @test_nowarn buffer(x, 10, N ÷ 2)
    @test length(y) == 2 * N - 1

    x = Timeseries(randn(101, 10), 0:0.1:10, 1:10)
    y = buffer(x, 10)
    @test length(y) == 10

    x = Timeseries(randn(100), 0.1:0.1:10)
    y = @test_nowarn window(x, 2, 1)
    @test all(length.(y) .== 2)
    y = @test_nowarn delayembed(x, 2, 1, 1)
    y = @test_nowarn delayembed(x, 2, 1, 2)
    @test length(y) == length(x)
    @test samplingperiod(y) == 2 * samplingperiod(x)
    y = @test_nowarn delayembed(x, 2, 2, 1)
end

@testitem "Rectification" tags=[:fast] begin
    import TimeseriesBase: rectifytime
    ts = 0.1:0.1:1000
    x = ToolsArray(sin, 𝑡(ts .+ randn(length(ts)) .* 1e-10))
    @test issorted(times(x))
    _x = @test_nowarn rectifytime(x)
    @test all(x .== _x)
    @test ts == times(_x)

    y = ToolsArray(cos, 𝑡(ts .+ randn(length(ts)) .* 1e-10))
    @test issorted(times(y))
    _x, _y = rectifytime(x, y)

    @test all(x .== parent(_x))
    @test ts == times(_x)
    @test all(y .== parent(_y))
    @test ts == times(_y)

    x = @test_nowarn Timeseries(randn(100, 10), 𝑡(1:100), X((1:10) .+ 1e-10 .* randn(10)))
    y = @test_nowarn rectify(x, dims = X)
    @test dims(y, X) == X(1:10)

    x = @test_nowarn Timeseries(randn(100, 10, 5), 𝑡(1:100),
                                X((1:10) .+ 1e-10 .* randn(10)),
                                Y((1:5) .+ 1e-10 .* randn(5)))
    y1 = @test_nowarn rectify(x, dims = X)
    y2 = @test_nowarn rectify(x, dims = Y)
    y3 = @test_nowarn rectify(x, dims = [X, Y])
    @test dims(y1, X) == dims(y3, X) == X(1:10)
    @test dims(y2, Y) == dims(y3, Y) == Y(1:5)
    @test dims(y1, Y) == dims(x, Y)
    @test dims(y2, X) == dims(x, X)
end

@testitem "Central differences" tags=[:fast] begin
    sig(n) = Timeseries(cumsum(randn(n)), range(0.01, 0.01, length = n))
    x = sig(1000)
    X = cat((sig(1000) for _ in 1:10)...; dims = Var(1:10))

    dx = @test_nowarn centraldiff(x)
    @test all(dx[2:(end - 1)] .== (parent(x)[3:end] - parent(x)[1:(end - 2)]) / 2)
    @test times(dx) == times(x)

    dX = @test_nowarn centraldiff(X)
    @test all(dX[2:(end - 1), :] .== (parent(X)[3:end, :] - parent(X)[1:(end - 2), :]) / 2)
    @test times(dX) == times(X)
    @test dims(dX, Var) == dims(X, Var)

    dX = @test_nowarn centralderiv(X)
    @test all(dX[2:(end - 1), :] .==
              ((parent(X)[3:end, :] - parent(X)[1:(end - 2), :]) / 2) ./ samplingperiod(X))
end

@testitem "Left and right derivatives" tags=[:fast] begin
    import TimeseriesBase: leftdiff, rightdiff
    sig(n) = Timeseries(cumsum(randn(n)), range(0.01, 0.01, length = n))
    x = sig(1000)
    X = cat((sig(1000) for _ in 1:10)...; dims = Var(1:10))

    dx = @test_nowarn leftdiff(x)
    @test all(parent(dx)[2:(end)] .== (parent(x)[2:end] - parent(x)[1:(end - 1)]))
    @test times(dx) == times(x)

    dX = @test_nowarn leftdiff(X)
    @test all(parent(dX)[2:(end), :] .== (parent(X)[2:end, :] - parent(X)[1:(end - 1), :]))
    @test times(dX) == times(X)
    @test dims(dX, Var) == dims(X, Var)

    dx = @test_nowarn rightdiff(x)
    @test all(parent(dx)[1:(end - 1)] .== (parent(x)[2:end] - parent(x)[1:(end - 1)]))
    @test times(dx) == times(x)

    dX = @test_nowarn rightdiff(X)
    @test all(parent(dX)[1:(end - 1), :] .==
              (parent(X)[2:end, :] - parent(X)[1:(end - 1), :]))
    @test times(dX) == times(X)
    @test dims(dX, Var) == dims(X, Var)
end

# @testitem "Irregular central derivative" begin
#     ts = 0.1:0.1:1000
#     x = Timeseries(ts, sin)
#     y = Timeseries(ts .+ randn(length(ts)) .* 1e-10, parent(x))
#     @test centralderiv(x) ≈ centralderiv(y)
# end

@testitem "Unitful derivative" tags=[:fast] begin
    using Unitful
    ts = 0.1:0.1:1000
    x = ToolsArray(sin, 𝑡(ts))
    y = set(x, 𝑡 => ts .* u"s")
    @test ustripall(centralderiv(x)) == ustripall(centralderiv(y))
    @test unit(eltype(centralderiv(y))) == unit(u"1/s")
end

@testitem "coarsegrain" tags=[:fast] begin
    using Statistics
    X = repeat(1:11, 1, 100)
    C = coarsegrain(X, dims = 1)
    M = mean(C, dims = 3)
    @test all(M[:, 1] .== 1.5:2:9.5)
    @test size(C, 1) == size(X, 1) ÷ 2

    C = coarsegrain(X)
    @test size(C) == (5, 50, 4)
    M = mean(C, dims = 3)
    @test all(M[:, 1] .== 1.5:2:9.5)

    C = coarsegrain(X; newdim = 2)
    M = mean(C, dims = 2)
    @test size(C) == (5, 200)
    @test all(M[:, 1] .== 1.5:2:9.5)

    X = cat(X, X; dims = 3)
    C = coarsegrain(X; dims = 1, newdim = 2)
    @test size(C) == (5, 200, 2)

    X = Timeseries(repeat(1:11, 1, 100), 1:11, 1:100)
    C = coarsegrain(X, dims = 1)
    M = dropdims(mean(C, dims = 3), dims = 3)
    @test all(M[:, 1] .== 1.5:2:9.5)
    @test size(C, 1) == size(X, 1) ÷ 2

    C = coarsegrain(X)
    @test size(C) == (5, 50, 4)
    M = dropdims(mean(C, dims = 3), dims = 3)
    @test all(M[:, 1] .== 1.5:2:9.5)

    C = coarsegrain(X; dims = 𝑡, newdim = Var)
    @test length(lookup(C, 1)) == size(C, 1)
    @test length(lookup(C, 2)) == size(C, 2)
    M = mean(C.data, dims = 2)
    @test size(C) == (5, 200)
    @test all(M[:, 1] .== 1.5:2:9.5)

    X = cat(X, X; dims = 3)
    C = coarsegrain(X; dims = 1, newdim = 2)
    @test size(C) == (5, 200, 2)
    @test_nowarn C[𝑡(Near(0.1))]
end

@testitem "matchdim" tags=[:fast] begin
    ts = 0:1:100
    X = [ToolsArray(sin, 𝑡(ts .+ 1e-6 .* randn(101))) for _ in 1:10]
    X = Timeseries(X, 1:10)
    Y = matchdim(X)

    @test length(unique(dims.(Y))) == 1
    @test dims(Y[1], 𝑡) == 𝑡(ts)
end

@testitem "regularize: single dimension" tags=[:fast] begin
    # Repair float jitter on a near-regular axis.
    ts = 0.1:0.1:100
    jittered = ts .+ randn(length(ts)) .* 1e-10
    d = 𝑡(jittered)
    grid, orig = regularize(d)
    @test grid isa AbstractRange
    @test length(grid) == length(jittered)
    @test orig == collect(jittered)         # genuine originals preserved
    # Recovered step is the rounded average, well within 1e-6 of true step.
    @test abs(step(grid) - 0.1) < 1e-6

    # Truly irregular input should throw under the default strict=true.
    irregular = collect(ts)
    irregular[50] += 5.0                     # one obvious outlier
    @test_throws ArgumentError regularize(𝑡(irregular))
    # strict=false downgrades to a warning and still returns a grid.
    g2, _ = @test_logs (:warn, r"not regular") regularize(𝑡(irregular);
                                                          strict = false)
    @test g2 isa AbstractRange

    # atol controls strictness; loose atol should make even jittered data pass
    # *and* tight atol on already-clean data should pass too.
    @test_nowarn regularize(𝑡(collect(0.0:0.5:10.0)); atol = 1e-12)
end

@testitem "regularize: AbstractDimArray" tags=[:fast] begin
    ts = 0.1:0.1:1000
    jit = ts .+ randn(length(ts)) .* 1e-10
    x = ToolsArray(sin, 𝑡(jit))
    y = @test_nowarn regularize(x)
    @test dims(y, 𝑡) isa typeof(dims(x, 𝑡))   # same Dimension type
    @test lookup(y, 𝑡) isa AbstractRange
    @test all(parent(y) .== parent(x))         # data untouched
    @test times(y) == ts

    # Multiple dims at once.
    z = Timeseries(randn(100, 10, 5), 𝑡(1:100),
                   X((1:10) .+ 1e-10 .* randn(10)),
                   Y((1:5) .+ 1e-10 .* randn(5)))
    w = @test_nowarn regularize(z; dims = [X, Y])
    @test dims(w, X) == X(1:10)
    @test dims(w, Y) == Y(1:5)
    @test dims(w, 𝑡) == dims(z, 𝑡)             # untouched dim stays put

    # zero=true: lookup is shifted to start at 0, genuine originals in metadata.
    ts2 = 100.5:0.25:200
    jit2 = ts2 .+ randn(length(ts2)) .* 1e-10
    a = ToolsArray(randn(length(ts2)), 𝑡(jit2))
    b = regularize(a; zero = true)
    @test first(times(b)) == 0
    @test lookup(b, 𝑡) isa AbstractRange
    md = metadata(b)
    @test haskey(md, :𝑡) || haskey(md, :t) || :𝑡 in keys(md)
    stored = md[first(intersect((:𝑡, :t), keys(md)))]
    @test stored == collect(jit2)              # originals, *not* a reconstruction
end

@testitem "regularize: multi-array alignment" tags=[:fast] begin
    using Statistics
    ts = 0:1:100
    Xs = [ToolsArray(sin, 𝑡(ts .+ 1e-6 .* randn(101))) for _ in 1:10]
    Ys = regularize(Xs)
    @test length(unique(dims.(Ys))) == 1
    @test dims(Ys[1], 𝑡) == 𝑡(0:1:100)

    # Vararg form is equivalent.
    a = ToolsArray(randn(50), 𝑡(collect(1.0:50.0) .+ 1e-9 .* randn(50)))
    b = ToolsArray(randn(50), 𝑡(collect(1.0:50.0) .+ 1e-9 .* randn(50)))
    ab = regularize(a, b)
    @test lookup(ab[1], 𝑡) == lookup(ab[2], 𝑡)
    @test lookup(ab[1], 𝑡) isa AbstractRange

    # Mismatched lengths: shared grid uses the minimum common length,
    # cropped to the overlapping range.
    long  = ToolsArray(randn(101), 𝑡(0.0:1.0:100.0))
    short = ToolsArray(randn(51), 𝑡(20.0:1.0:70.0))
    aligned = regularize([long, short])
    @test length(unique(size.(aligned, 𝑡))) == 1
    @test first(times(aligned[1])) == 20.0
    @test last(times(aligned[1])) == 70.0

    # Non-overlapping ranges throw.
    far_a = ToolsArray(randn(10), 𝑡(0.0:1.0:9.0))
    far_b = ToolsArray(randn(10), 𝑡(100.0:1.0:109.0))
    @test_throws ArgumentError regularize([far_a, far_b])
end

@testitem "regularize: Unitful" tags=[:fast] begin
    using Unitful
    ts = (1:1000)u"s"
    jittered = ts .+ randn(1000) .* 1e-10u"s"
    x = ToolsArray(randn(1000), 𝑡(jittered))
    y = @test_nowarn regularize(x)
    @test unit(eltype(lookup(y, 𝑡))) == u"s"
    @test lookup(y, 𝑡) isa AbstractRange
    @test abs(step(lookup(y, 𝑡)) - 1.0u"s") < 1e-6u"s"
end

@testitem "regularize: deprecation forwarders" tags=[:fast] begin
    # rectify / rectifytime / matchdim still work, just route through
    # regularize. The behaviour assertions are the same as before.
    ts = 0.1:0.1:100
    x = ToolsArray(sin, 𝑡(ts .+ randn(length(ts)) .* 1e-10))
    y = rectify(x; dims = 𝑡)
    @test lookup(y, 𝑡) isa AbstractRange
    @test times(y) == ts

    y2 = rectifytime(x)
    @test times(y2) == ts

    Xs = [ToolsArray(sin, 𝑡(0.0:1.0:100.0 .+ 1e-6 .* randn(101))) for _ in 1:3]
    Ys = matchdim(Xs)
    @test length(unique(dims.(Ys))) == 1
end

@testitem "regularize: LSQ fit, noisy endpoints" tags=[:fast] begin
    # The LSQ fit must not be dominated by the first or last sample. Construct
    # a clean grid, displace both endpoints in opposite directions, and check
    # the fitted step still matches the true step within the displacement size.
    n = 1001
    true_step = 0.1
    clean = collect(0.0:true_step:(n - 1) * true_step)
    noisy = copy(clean)
    noisy[1]   -= 0.01
    noisy[end] += 0.01
    grid, _ = regularize(𝑡(noisy); atol = 0.02, strict = false)
    # Endpoint-pinning would give step = (clean[end] + 0.01 - (clean[0] - 0.01)) / (n-1)
    # = true_step + 0.02/(n-1) ≈ 0.1 + 2e-5; LSQ should be ~1000x closer.
    @test abs(step(grid) - true_step) < 1e-6
end

@testitem "regularize: already-regular fast path" tags=[:fast] begin
    # If the input is already an AbstractRange, regularize is a no-op pass-through.
    r = 0.0:0.5:50.0
    grid, _ = regularize(𝑡(r))
    @test grid === r || grid == r
    @test step(grid) == 0.5

    x = ToolsArray(randn(101), 𝑡(r))
    y = regularize(x)
    @test lookup(y, 𝑡) == r
end

@testitem "regularize: unsorted and degenerate inputs" tags=[:fast] begin
    # Unsorted input is rejected up-front (rather than silently producing a
    # nonsense fit).
    unsorted = [0.0, 0.1, 0.3, 0.2, 0.4]
    @test_throws ArgumentError regularize(𝑡(unsorted))

    # Fewer than 2 points is not enough to define a grid.
    @test_throws ArgumentError regularize(𝑡([1.0]))
end

@testitem "regularize: round-to-atol is scale-invariant" tags=[:fast] begin
    # Rounding behaviour shouldn't depend on the magnitude of the step. A
    # near-0.1 step and a near-1e6 step should both round cleanly when
    # `atol` is set relative to the step.
    small = collect(0.0:0.1:10.0) .+ 1e-12 .* randn(101)
    g_small, _ = regularize(𝑡(small); atol = 1e-8)
    @test step(g_small) == 0.1

    large = collect(0.0:1e6:1e8) .+ 1.0 .* randn(101)
    g_large, _ = regularize(𝑡(large); atol = 10.0)
    @test step(g_large) == 1e6
end
