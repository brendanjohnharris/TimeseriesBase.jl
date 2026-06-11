@testitem "Interlace" tags = [:fast] begin
    x = Timeseries(randn(11), 0:0.1:1)
    y = Timeseries(randn(10), 0.05:0.1:1)
    z = @test_nowarn interlace(x, y)
    @test all(collect(times(z)) .== 0.0:0.05:1.0)

    # interlace is univariate-only; multivariate input is rejected rather than
    # silently producing a wrongly-indexed univariate result.
    @test_throws MethodError interlace(
        Timeseries(randn(11, 3), 0:0.1:1, Var(1:3)),
        Timeseries(randn(10, 3), 0.05:0.1:1, Var(1:3))
    )
end

@testitem "Cat and stack" tags = [:fast] begin
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

@testitem "Buffer" tags = [:fast] begin
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

    # Window longer than the series: no complete buffers. Returns empty rather
    # than throwing a BoundsError (discard=true is the default).
    xs = Timeseries(randn(5), 0.1:0.1:0.5)
    @test length(@test_nowarn buffer(xs, 10)) == 0
    @test length(buffer(xs, 10, 0; discard = false)) == 1
end

@testitem "Rectification" tags = [:fast] begin
    import TimeseriesBase: rectifytime
    ts = 0.1:0.1:1000
    x = ToolsArray(sin, 𝑡(ts .+ randn(length(ts)) .* 1.0e-10))
    @test issorted(times(x))
    _x = @test_nowarn rectifytime(x)
    @test all(x .== _x)
    @test ts == times(_x)

    y = ToolsArray(cos, 𝑡(ts .+ randn(length(ts)) .* 1.0e-10))
    @test issorted(times(y))
    _x, _y = rectifytime(x, y)

    @test all(x .== parent(_x))
    @test ts == times(_x)
    @test all(y .== parent(_y))
    @test ts == times(_y)

    x = @test_nowarn Timeseries(randn(100, 10), 𝑡(1:100), X((1:10) .+ 1.0e-10 .* randn(10)))
    y = @test_nowarn rectify(x, dims = X)
    @test dims(y, X) == X(1:10)

    x = @test_nowarn Timeseries(
        randn(100, 10, 5), 𝑡(1:100),
        X((1:10) .+ 1.0e-10 .* randn(10)),
        Y((1:5) .+ 1.0e-10 .* randn(5))
    )
    y1 = @test_nowarn rectify(x, dims = X)
    y2 = @test_nowarn rectify(x, dims = Y)
    y3 = @test_nowarn rectify(x, dims = [X, Y])
    @test dims(y1, X) == dims(y3, X) == X(1:10)
    @test dims(y2, Y) == dims(y3, Y) == Y(1:5)
    @test dims(y1, Y) == dims(x, Y)
    @test dims(y2, X) == dims(x, X)
end

@testitem "Central differences" tags = [:fast] begin
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
    @test all(
        dX[2:(end - 1), :] .==
            ((parent(X)[3:end, :] - parent(X)[1:(end - 2), :]) / 2) ./ samplingperiod(X)
    )
end

@testitem "Left and right derivatives" tags = [:fast] begin
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
    @test all(
        parent(dX)[1:(end - 1), :] .==
            (parent(X)[2:end, :] - parent(X)[1:(end - 1), :])
    )
    @test times(dX) == times(X)
    @test dims(dX, Var) == dims(X, Var)
end

# @testitem "Irregular central derivative" begin
#     ts = 0.1:0.1:1000
#     x = Timeseries(ts, sin)
#     y = Timeseries(ts .+ randn(length(ts)) .* 1e-10, parent(x))
#     @test centralderiv(x) ≈ centralderiv(y)
# end

@testitem "Unitful derivative" tags = [:fast] begin
    using Unitful
    ts = 0.1:0.1:1000
    x = ToolsArray(sin, 𝑡(ts))
    y = set(x, 𝑡 => ts .* u"s")
    @test ustripall(centralderiv(x)) == ustripall(centralderiv(y))
    @test unit(eltype(centralderiv(y))) == unit(u"1/s")
end

@testitem "coarsegrain" tags = [:fast] begin
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

@testitem "matchdim" tags = [:fast] begin
    ts = 0:1:100
    X = [ToolsArray(sin, 𝑡(ts .+ 1.0e-6 .* randn(101))) for _ in 1:10]
    X = Timeseries(X, 1:10)
    Y = matchdim(X)

    @test length(unique(dims.(Y))) == 1
    @test dims(Y[1], 𝑡) == 𝑡(ts)
end

@testitem "regularize: single dimension" tags = [:fast] begin
    # Repair float jitter on a near-regular axis.
    ts = 0.1:0.1:100
    jittered = ts .+ randn(length(ts)) .* 1.0e-10
    d = 𝑡(jittered)
    grid, orig = regularize(d)
    @test grid isa AbstractRange
    @test length(grid) == length(jittered)
    @test orig == collect(jittered)         # genuine originals preserved
    # Recovered step is the rounded average, well within 1e-6 of true step.
    @test abs(step(grid) - 0.1) < 1.0e-6

    # Truly irregular input should throw under the default strict=true.
    # Displace the last point so the step is wildly irregular while the axis
    # stays sorted (an interior jump that big would break monotonicity, which
    # is a separate, hard-thrown precondition).
    irregular = collect(ts)
    irregular[end] += 5.0                    # one obvious outlier
    @test_throws ArgumentError regularize(𝑡(irregular))
    # strict=false downgrades to a warning and still returns a grid.
    g2, _ = @test_logs (:warn, r"not regular") regularize(
        𝑡(irregular);
        strict = false
    )
    @test g2 isa AbstractRange

    # atol controls strictness; loose atol should make even jittered data pass
    # *and* tight atol on already-clean data should pass too.
    @test_nowarn regularize(𝑡(collect(0.0:0.5:10.0)); atol = 1.0e-12)
end

@testitem "regularize: AbstractDimArray" tags = [:fast] begin
    ts = 0.1:0.1:1000
    jit = ts .+ randn(length(ts)) .* 1.0e-10
    x = ToolsArray(sin, 𝑡(jit))
    y = @test_nowarn regularize(x)
    # Same Dimension wrapper; the lookup type changes (Irregular → Regular).
    @test dims(y, 𝑡) isa 𝑡
    @test parent(lookup(y, 𝑡)) isa AbstractRange
    @test all(parent(y) .== parent(x))         # data untouched
    @test times(y) == ts

    # Multiple dims at once.
    z = Timeseries(
        randn(100, 10, 5), 𝑡(1:100),
        X((1:10) .+ 1.0e-10 .* randn(10)),
        Y((1:5) .+ 1.0e-10 .* randn(5))
    )
    w = @test_nowarn regularize(z; dims = [X, Y])
    @test dims(w, X) == X(1:10)
    @test dims(w, Y) == Y(1:5)
    @test dims(w, 𝑡) == dims(z, 𝑡)             # untouched dim stays put

    # zero=true: lookup is shifted to start at 0, genuine originals in metadata.
    ts2 = 100.5:0.25:200
    jit2 = ts2 .+ randn(length(ts2)) .* 1.0e-10
    a = ToolsArray(randn(length(ts2)), 𝑡(jit2))
    b = regularize(a; zero = true)
    @test first(times(b)) == 0
    @test parent(lookup(b, 𝑡)) isa AbstractRange
    md = metadata(b)
    @test haskey(md, :𝑡) || haskey(md, :t) || :𝑡 in keys(md)
    stored = md[first(intersect((:𝑡, :t), keys(md)))]
    @test stored == collect(jit2)              # originals, *not* a reconstruction
end

@testitem "regularize: multi-array alignment" tags = [:fast] begin
    using Statistics
    ts = 0:1:100
    Xs = [ToolsArray(sin, 𝑡(ts .+ 1.0e-8 .* randn(101))) for _ in 1:10]
    Ys = regularize(Xs)
    @test length(unique(dims.(Ys))) == 1
    @test dims(Ys[1], 𝑡) == 𝑡(0:1:100)

    # Vararg form is equivalent.
    a = ToolsArray(randn(50), 𝑡(collect(1.0:50.0) .+ 1.0e-9 .* randn(50)))
    b = ToolsArray(randn(50), 𝑡(collect(1.0:50.0) .+ 1.0e-9 .* randn(50)))
    ab = regularize(a, b)
    @test lookup(ab[1], 𝑡) == lookup(ab[2], 𝑡)
    @test parent(lookup(ab[1], 𝑡)) isa AbstractRange

    # Mismatched lengths: shared grid uses the minimum common length,
    # cropped to the overlapping range.
    long = ToolsArray(randn(101), 𝑡(0.0:1.0:100.0))
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

@testitem "regularize: Unitful" tags = [:fast] begin
    using Unitful
    ts = (1:1000)u"s"
    jittered = ts .+ randn(1000) .* 1.0e-10u"s"
    x = ToolsArray(randn(1000), 𝑡(jittered))
    y = @test_nowarn regularize(x)
    @test unit(eltype(lookup(y, 𝑡))) == u"s"
    @test parent(lookup(y, 𝑡)) isa AbstractRange
    @test abs(step(lookup(y, 𝑡)) - 1.0u"s") < 1.0e-6u"s"
end

@testitem "regularize: LSQ fit, noisy endpoints" tags = [:fast] begin
    # The LSQ fit must not be dominated by the first or last sample. Construct
    # a clean grid, displace both endpoints in opposite directions, and check
    # the fitted step still matches the true step within the displacement size.
    n = 1001
    true_step = 0.1
    clean = collect(0.0:true_step:((n - 1) * true_step))
    noisy = copy(clean)
    noisy[1] -= 0.01
    noisy[end] += 0.01
    grid, _ = regularize(𝑡(noisy); atol = 0.02, strict = false)
    # Endpoint-pinning would give step = (clean[end] + 0.01 - (clean[0] - 0.01)) / (n-1)
    # = true_step + 0.02/(n-1) ≈ 0.1 + 2e-5; LSQ should be ~1000x closer.
    @test abs(step(grid) - true_step) < 1.0e-6
end

@testitem "regularize: already-regular fast path" tags = [:fast] begin
    # If the input is already an AbstractRange, regularize is a no-op pass-through.
    r = 0.0:0.5:50.0
    grid, _ = regularize(𝑡(r))
    @test grid === r || grid == r
    @test step(grid) == 0.5

    x = ToolsArray(randn(101), 𝑡(r))
    y = regularize(x)
    @test lookup(y, 𝑡) == r
end

@testitem "regularize: unsorted and degenerate inputs" tags = [:fast] begin
    # Unsorted input is rejected up-front (rather than silently producing a
    # nonsense fit).
    unsorted = [0.0, 0.1, 0.3, 0.2, 0.4]
    @test_throws ArgumentError regularize(𝑡(unsorted))

    # Fewer than 2 points is not enough to define a grid.
    @test_throws ArgumentError regularize(𝑡([1.0]))
end

@testitem "regularize: round-to-atol is scale-invariant" tags = [:fast] begin
    # Rounding behaviour shouldn't depend on the magnitude of the step. A
    # near-0.1 step and a near-1e6 step should both round cleanly when
    # `atol` is set relative to the step.
    small = collect(0.0:0.1:10.0) .+ 1.0e-12 .* randn(101)
    g_small, _ = regularize(𝑡(small); atol = 1.0e-8)
    @test step(g_small) == 0.1

    large = collect(0.0:1.0e6:1.0e8) .+ 1.0 .* randn(101)
    g_large, _ = regularize(𝑡(large); atol = 10.0)
    @test step(g_large) == 1.0e6
end

@testitem "Spike trains" tags = [:fast] begin
    spike_times = [1.5, 3.2, 0.8, 5.1]
    st = @test_nowarn spiketrain(spike_times)
    @test st isa SpikeTrain
    @test st isa UnivariateSpikeTrain
    @test all(parent(st))                       # every stored sample is a spike
    @test times(st) == sort(spike_times)        # sorted on construction
    @test spiketimes(st) == sort(spike_times)   # round-trips

    # spiketimes on a plain array is the identity.
    @test spiketimes([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    # Multivariate: one spike-time vector per channel.
    mt = Timeseries(Bool[1 0; 0 1; 1 1], 𝑡(1.0:3.0), Var(1:2))
    smt = spiketimes(mt)
    @test length(smt) == 2
    @test smt[1] == [1.0, 3.0]
    @test smt[2] == [2.0, 3.0]
end

@testitem "align" tags = [:fast] begin
    x = Timeseries(collect(1.0:100.0), 𝑡(1.0:100.0))
    a = @test_nowarn align(x, [20.0, 50.0, 80.0], [-2.0, 2.0])
    @test length(a) == 3
    @test times(a) == [20.0, 50.0, 80.0]

    # zero=true (default): each window's lookup is centred on its trigger, while
    # the data values stay put.
    w = a[2]
    @test length(w) == 5
    @test collect(times(w)) == -2.0:1.0:2.0
    @test collect(w) == [48.0, 49.0, 50.0, 51.0, 52.0]

    # zero=false keeps the absolute times.
    b = align(x, [50.0], [-2.0, 2.0]; zero = false)
    @test collect(times(b[1])) == 48.0:1.0:52.0

    # Interval form is equivalent to the tuple form.
    c = align(x, [50.0], -2.0 .. 2.0)
    @test collect(times(c[1])) == collect(times(a[2]))
end

@testitem "stitch" tags = [:fast] begin
    @test :stitch in names(TimeseriesBase)   # exported at the top level

    x = Timeseries(collect(1.0:10.0), 𝑡(0.1:0.1:1.0))
    y = Timeseries(collect(11.0:20.0), 𝑡(0.1:0.1:1.0))
    z = @test_nowarn stitch(x, y)
    @test z isa UnivariateRegular
    @test parent(z) == 1.0:20.0
    @test samplingperiod(z) == samplingperiod(x)
    @test times(z) == 0.1:0.1:2.0

    # Multivariate: concatenate along time, keep the other dimensions.
    X = Timeseries(reshape(collect(1.0:20.0), 10, 2), 𝑡(0.1:0.1:1.0), Var(1:2))
    Y = Timeseries(reshape(collect(21.0:40.0), 10, 2), 𝑡(0.1:0.1:1.0), Var(1:2))
    Z = stitch(X, Y)
    @test size(Z) == (20, 2)
    @test dims(Z, Var) == Var(1:2)

    # Vararg reduce form.
    @test length(stitch(x, y, x)) == 30
end

@testitem "Circular statistics" tags = [:fast] begin
    # Uniformly spread phases: ~zero resultant length, unit circular variance.
    θ = range(0, 2π, length = 13)[1:12]
    @test resultantlength(θ) < 1.0e-10
    @test circularvar(θ) ≈ 1 atol = 1.0e-10

    # Concentrated phases: resultant length near 1, mean near the cluster centre.
    ϕ = [0.01, -0.01, 0.0, 0.02, -0.02]
    @test resultantlength(ϕ) > 0.999
    @test circularmean(ϕ) ≈ 0 atol = 1.0e-3
    @test circularvar(ϕ) ≈ 1 - resultantlength(ϕ)
    @test circularstd(ϕ) ≈ sqrt(-2 * log(resultantlength(ϕ)))

    # circularmean wraps: phases either side of ±π average to π, not 0.
    @test abs(circularmean([π - 0.01, -π + 0.01])) ≈ π atol = 1.0e-2
end

@testitem "phasegrad" tags = [:fast] begin
    @test phasegrad(0.1, 0.0) ≈ 0.1
    @test phasegrad(0.0, 0.1) ≈ -0.1
    # Crossing the 2π boundary gives a small wrapped difference, not ~2π.
    @test phasegrad(0.1, 2π - 0.1) ≈ 0.2 atol = 1.0e-10
    @test phasegrad(2π - 0.1, 0.1) ≈ -0.2 atol = 1.0e-10
    # Complex inputs use their angle.
    @test phasegrad(exp(im * 0.1), exp(im * 0.0)) ≈ 0.1
    # Vectorised over arrays.
    @test phasegrad([0.1, 0.2], [0.0, 0.0]) ≈ [0.1, 0.2]
end

@testitem "Metadata helpers" tags = [:fast] begin
    import DimensionalData as DD

    # addmetadata on an array with no existing metadata (regression: this used
    # to throw a MethodError on NoMetadata).
    x = Timeseries(randn(10), 𝑡(1:10))
    @test DD.metadata(x) isa DD.NoMetadata
    x2 = @test_nowarn addmetadata(x; foo = 1, bar = 2)
    @test DD.metadata(x2)[:foo] == 1
    @test DD.metadata(x2)[:bar] == 2

    # Merges with existing metadata.
    x3 = Timeseries(randn(10), 𝑡(1:10); metadata = Dict(:a => 1))
    x4 = addmetadata(x3; b = 2)
    @test DD.metadata(x4)[:a] == 1
    @test DD.metadata(x4)[:b] == 2

    # Warns and overwrites on a duplicate key.
    x5 = @test_logs (:warn, r"already contains") addmetadata(x3; a = 99)
    @test DD.metadata(x5)[:a] == 99

    # addrefdim appends a reference dimension.
    xr = addrefdim(x, Var(1))
    @test Var(1) in DD.refdims(xr)
end

@testitem "nyquist" tags = [:fast] begin
    using Unitful
    rts = Timeseries(randn(1001), 0:0.01:10)
    @test nyquist(rts) == samplingrate(rts) / 2
    @test nyquist(rts) == 50.0

    # Unitful: the Nyquist frequency carries the (inverse-time) units of the rate.
    uts = Timeseries(randn(100), (0:0.01:0.99)u"s")
    @test nyquist(uts) == samplingrate(uts) / 2
    @test dimension(nyquist(uts)) == dimension(u"Hz")
end
