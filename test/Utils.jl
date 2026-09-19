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

    # Endpoints repeat their neighbours' central differences rather than switching to a
    # one-sided stencil; the docstring says so, so pin it.
    @test parent(dx)[1] == parent(dx)[2]
    @test parent(dx)[end] == parent(dx)[end - 1]
    @test parent(centraldiff(X))[1, :] == parent(centraldiff(X))[2, :]
    @test parent(centraldiff(X))[end, :] == parent(centraldiff(X))[end - 1, :]
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

@testitem "Differences act along dims, not linear indices" tags = [:fast] begin
    import TimeseriesBase: leftdiff, rightdiff
    D = [1.0 10.0; 2.0 20.0; 4.0 40.0; 8.0 80.0]
    X = Timeseries(copy(D), 0.0:0.5:1.5, [:a, :b])
    for f in (centraldiff, leftdiff, rightdiff, centralderiv, leftderiv, rightderiv)
        # Each column must match the univariate result for that column, boundaries included;
        # a kernel that indexes the flattened parent leaks across the column boundary.
        expected = hcat((parent(f(Timeseries(D[:, j], 0.0:0.5:1.5))) for j in axes(D, 2))...)
        @test parent(f(X)) == expected
    end
end

@testitem "regularize: atol is converted to the lookup's unit" tags = [:fast] begin
    using Unitful
    x = ToolsArray(randn(10), (𝑡((0.0:1.0:9.0)u"s"),))
    # 100 ms and 0.1 s are the same tolerance and must give the same grid
    @test collect(times(regularize(x; atol = 100u"ms"))) ==
        collect(times(regularize(x; atol = 0.1u"s")))
    @test collect(times(regularize(x; atol = 0.1u"s"))) ≈ collect((0.0:1.0:9.0)u"s")
end

@testitem "regularize: returned grid is within atol of the input" tags = [:fast] begin
    for n in (100, 1000, 10000)
        t = collect((0:(n - 1)) .* (1 / 3)) # exactly regular, not decimal-friendly
        x = ToolsArray(randn(n), (𝑡(t),))
        @test maximum(abs.(collect(times(regularize(x))) .- t)) <= 1.0e-6 * (1 / 3)
    end
end

@testitem "buffer labels each buffer with its own centre" tags = [:fast] begin
    using Statistics
    t = 1.0:10.0
    x = Timeseries(collect(t), t) # value == time, so the centre is computable
    for discard in (true, false)
        b = buffer(x, 3; discard)
        @test collect(times(b)) ≈ [mean(parent(bb)) for bb in b]
    end
end

@testitem "delayembed delays match the actual sample spacing" tags = [:fast] begin
    t = 1.0:30.0
    x = Timeseries(collect(t), t) # value == time
    for (n, τ, p) in ((3, 2, 1), (3, 2, 2), (4, 3, 3), (3, 1, 2))
        e = delayembed(x, n, τ, p)
        delays = collect(lookup(e, 2))
        heads = collect(lookup(e, 1))
        @test all(parent(e)[i, :] ≈ heads[i] .+ delays for i in axes(e, 1))
    end
end

@testitem "circular statistics stay in their documented ranges" tags = [:fast] begin
    using Random
    Random.seed!(1)
    # Identical angles give |resultant| == 1 up to float error; one ulp over sends
    # circularstd's log positive and its sqrt into DomainError.
    θs = [fill(rand() * 2π, rand(2:8)) for _ in 1:2000]
    @test all(resultantlength(θ) <= 1 for θ in θs)
    @test all(0 <= circularvar(θ) <= 1 for θ in θs)
    @test all(circularstd(θ) >= 0 for θ in θs)
    @test size(circularvar(rand(4, 3); dims = 1)) == (1, 3) # `dims` is documented
end

@testitem "rectify handles a descending lookup" tags = [:fast] begin
    import TimeseriesBase: rectify
    t = collect(9.0:-1.0:0.0)
    ts, _ = rectify(𝑡(t))
    @test collect(ts) ≈ t
end

@testitem "regularize zero=true keeps the original lookup" tags = [:fast] begin
    t = collect(0.0:1.0:9.0)
    # metadata already carrying the dim's own name must not displace the original
    x = ToolsArray(randn(10), (𝑡(t),); metadata = Dict(:𝑡 => "pre-existing"))
    @test metadata(regularize(x; zero = true))[:𝑡] == t
    xs = [ToolsArray(randn(10), (𝑡(t),); metadata = Dict(:𝑡 => "pre-existing")) for _ in 1:2]
    @test metadata(regularize(xs; zero = true)[1])[:𝑡] == t
end

@testitem "regularize reports single-sample inputs clearly" tags = [:fast] begin
    a = ToolsArray(randn(1), (𝑡([1.0]),))
    b = ToolsArray(randn(1), (𝑡([1.0]),))
    e = try
        regularize(a, b)
        nothing
    catch e
        e
    end
    @test e isa ArgumentError
    @test occursin("at least 2", e.msg) # not "median of an empty array"
end

@testitem "align accepts a dimension as well as an index" tags = [:fast] begin
    x = Timeseries(randn(20), 0.0:1.0:19.0)
    @test align(x, [5.0, 10.0], (-2.0, 2.0); dims = 𝑡) == align(x, [5.0, 10.0], (-2.0, 2.0))
    @test_throws ArgumentError align(x, [5.0], (-2.0, 2.0); dims = (𝑡, Var))
end

@testitem "rectify accepts an integer dims" tags = [:fast] begin
    import TimeseriesBase: rectify
    a = ToolsArray(randn(10), (𝑡(collect(0.0:1.0:9.0)),))
    b = ToolsArray(randn(10), (𝑡(collect(0.0:1.0:9.0)),))
    @test collect(times(rectify(a, b; dims = 1)[1])) ≈ collect(times(rectify(a, b; dims = 𝑡)[1]))
end

@testitem "selectors accept an unformatted dimension" tags = [:fast] begin
    import TimeseriesBase.Utils: At, Near
    @test At(𝑡(1:3)) == At(1:3) # bare dim: `val` is the range itself, not a Lookup
    x = ToolsArray(randn(3), (𝑡(1:3),))
    @test At(dims(x, 𝑡)) == At(1:3) # formatted dim: `val` is a Lookup
    @test Near(𝑡(1:3)) == Near(1:3)
end

@testitem "Dropdims requires dims" tags = [:fast] begin
    x = randn(3, 4)
    @test Dropdims(sum)(x; dims = 1) == dropdims(sum(x; dims = 1); dims = 1)
    @test_throws UndefKeywordError Dropdims(sum)(x)
end

@testitem "Dates: rate accessors" tags = [:fast] begin
    using Dates, Unitful
    x = Timeseries(cumsum(randn(100)), DateTime(2020, 1, 1):Day(1):DateTime(2020, 4, 9))
    @test samplingperiod(x) == Day(1)          # unchanged: a Period
    @test samplingrate(x) == 1 / (86400 * u"s") # a genuine rate
    @test nyquist(x) == samplingrate(x) / 2
    @test timeunit(x) == NoUnits               # no Unitful unit, but must not throw

    h = Timeseries(randn(10), DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 1, 9))
    @test samplingrate(h) == 1 / (3600 * u"s")

    # A calendar period has no fixed length, so a rate is undefined; say so clearly.
    y = Timeseries(collect(1:100), DateTime(1901):Year(1):DateTime(2000))
    @test_throws ArgumentError samplingrate(y)
    @test_throws ArgumentError nyquist(y)
    @test samplingperiod(y) == Year(1)         # still fine
    @test duration(y) == DateTime(2000) - DateTime(1901)
end

@testitem "Dates: derivatives" tags = [:fast] begin
    using Dates, Unitful
    t = DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 10)
    x = Timeseries(cumsum(randn(10)), t)
    for (d, dv) in ((centralderiv, centraldiff), (leftderiv, leftdiff), (rightderiv, rightdiff))
        y = d(x)
        @test unit(eltype(y)) == u"s^-1"
        @test ustripall(y) ≈ parent(dv(x)) ./ 86400
        @test times(y) == times(x)             # the Dates axis is preserved
    end
    # In place cannot work: the result carries rate units the input cannot store.
    @test_throws ArgumentError centralderiv!(deepcopy(x))
    # A calendar period still has no rate.
    @test_throws ArgumentError centralderiv(Timeseries(randn(10), DateTime(1901):Year(1):DateTime(1910)))
end

@testitem "MultidimensionalTimeseries admits mixed lookup types" tags = [:fast] begin
    @test Timeseries(randn(8, 3, 3), 1:8, 𝑥(1:3), 𝑦(1.0:3.0)) isa MultidimensionalTimeseries
    @test Timeseries(randn(8, 3, 3), 1:8, 𝑥(1:3), 𝑦(1:3)) isa MultidimensionalTimeseries
    @test Timeseries(randn(8, 3), 1:8, 𝑥(1:3)) isa MultidimensionalTimeseries
    # An irregular dimension, time or otherwise, must still be excluded.
    @test !(Timeseries(randn(8, 3), 1:8, 𝑥([1.0, 2.0, 4.0])) isa MultidimensionalTimeseries)
    @test !(Timeseries(randn(8, 3), collect(1.0:8.0) .^ 2, 𝑥(1:3)) isa MultidimensionalTimeseries)
end

@testitem "phasegrad wraps to [-π, π)" tags = [:fast] begin
    @test phasegrad(float(π), 0.0) ≈ -π # closed at the lower end, as documented
    @test phasegrad(0.1, 2π - 0.1) ≈ 0.2
    a = rand(500) .* 4π .- 2π
    b = rand(500) .* 4π .- 2π
    @test all(-π .<= phasegrad.(a, b) .< π)
end

@testitem "Dates: windowing" tags = [:fast] begin
    using Dates, Statistics
    t = DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 30)
    x = Timeseries(randn(30), t)

    b = buffer(x, 5)
    @test length(b) == 6
    @test times(b) isa AbstractRange   # full buffers sit on a regular grid
    @test eltype(times(b)) == DateTime
    @test first(times(b)) == DateTime(2020, 1, 3) # centre of samples 1:5

    # A short final buffer sits off that grid, so centres are labelled per-buffer.
    b2 = buffer(x, 7, 0; discard = false)
    @test !(times(b2) isa AbstractRange)
    @test eltype(times(b2)) == DateTime

    @test size(window(x, 5, 2)) == size(window(Timeseries(randn(30), 1.0:1.0:30.0), 5, 2))

    e = delayembed(x, 3, 2)
    @test lookup(e, :delay) == [Day(-4), Day(-2), Day(0)] # `zero(δt)`, not `0`
    @test eltype(lookup(e, 𝑡)) == DateTime

    c = coarsegrain(x; dims = 𝑡)
    @test times(c)[1] == DateTime(2020, 1, 1, 12)
    @test times(c)[2] == DateTime(2020, 1, 3, 12)

    # A `Date` axis has day resolution, not millisecond; a `Period` axis is a vector
    # space but integer-quantised, so `mean([Day(1), Day(2)])` throws on its own.
    d = Timeseries(randn(20), Date(2020, 1, 1):Day(1):Date(2020, 1, 20))
    @test eltype(times(buffer(d, 4))) == Date
    # Samples 1:4 have mean offset 1.5 days, which rounds to even.
    @test first(times(buffer(d, 4))) == Date(2020, 1, 3)

    p = Timeseries(randn(20), Day(1):Day(1):Day(20))
    @test eltype(times(buffer(p, 4))) == Day

    # Window longer than the series: an empty result must not go through
    # `times(x)[1:0]`, which needs `copysign(::Day, ::Day)`.
    @test length(@test_nowarn buffer(Timeseries(randn(3), t[1:3]), 10)) == 0
    @test length(@test_nowarn buffer(d, 40)) == 0
    @test times(coarsegrain(p; dims = 𝑡))[1] == Day(1)

    # The numeric path is untouched.
    n = Timeseries(randn(30), 1.0:1.0:30.0)
    @test times(buffer(n, 5)) == 3.0:5.0:28.0
    @test times(coarsegrain(n; dims = 𝑡))[1] == 1.5
end

@testitem "Dates: grid repair" tags = [:fast] begin
    using Dates
    t = DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 30)
    x = Timeseries(randn(30), t)

    # An exact Dates range is exact by construction: returned untouched, step unit and
    # all, rather than relabelled as milliseconds.
    @test times(regularize(x)) === t
    @test step(times(regularize(x))) == Day(1)

    d = Timeseries(randn(20), Date(2020, 1, 1):Day(1):Date(2020, 1, 20))
    @test times(regularize(d)) == times(d)

    # A genuinely irregular lookup. The default tolerance is exact, because a Dates
    # lookup carries no float jitter for a loose default to absorb.
    tv = collect(t)
    tv[5] += Hour(3)
    xi = Timeseries(randn(30), tv)
    @test_throws ArgumentError regularize(xi)
    @test times(regularize(xi; atol = Hour(6))) isa AbstractRange
    @test_throws ArgumentError regularize(xi; atol = Minute(1))

    # `zero = true` means elapsed from the origin, so it yields a uniform-unit Period
    # range; the genuine original lookup goes to metadata.
    z = regularize(x; zero = true)
    @test eltype(times(z)) == Millisecond
    @test first(times(z)) == Millisecond(0)
    @test step(times(z)) == Millisecond(86_400_000)
    @test metadata(z)[:𝑡][1] == DateTime(2020, 1, 1)
    @test eltype(times(regularize(d; zero = true))) == Day

    # Several arrays onto one grid: the path that replaces `matchdim`.
    m = regularize([x, x])
    @test times(m[1]) == times(m[2])
    @test eltype(times(m[1])) == DateTime

    # The superseded trio is rejected rather than ported.
    @test_throws ArgumentError rectify(x; dims = 𝑡)
    @test_throws ArgumentError rectifytime(x)
    @test_throws ArgumentError matchdim([x, x])

    # The numeric path is untouched.
    n = Timeseries(randn(30), 1.0:1.0:30.0)
    @test times(regularize(n)) == 1.0:1.0:30.0
    @test times(regularize(n; zero = true)) == 0.0:1.0:29.0
    @test times(rectify(n; dims = 𝑡)) == 1.0:1.0:30.0
end

@testitem "Dates: stitch keeps the calendar" tags = [:fast] begin
    using Dates
    t = DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 30)
    x = Timeseries(randn(30), t)
    s = stitch(x, x)
    @test length(s) == 60
    @test first(times(s)) == DateTime(2020, 1, 1) # not Day(1)
    @test step(times(s)) == Day(1)
    @test last(times(s)) == DateTime(2020, 2, 29) # 2020 is a leap year
    @test eltype(times(s)) == DateTime

    X = Timeseries(randn(30, 3), t, [:a, :b, :c])
    @test first(times(stitch(X, X))) == DateTime(2020, 1, 1)
    @test size(stitch(X, X)) == (60, 3)

    # An elapsed axis is already elapsed, so it restarts one step in as before.
    p = Timeseries(randn(20), Day(1):Day(1):Day(20))
    @test first(times(stitch(p, p))) == Day(1)

    n = Timeseries(randn(30), 1.0:1.0:30.0)
    @test times(stitch(n, n)) == 1.0:1.0:60.0
end

@testitem "Dates: unsupported lookup types" tags = [:fast] begin
    using Dates
    # `Time` is a time of day: it wraps at midnight and its ranges are unreliable in
    # Base, so it is rejected rather than half-supported.
    tt = Timeseries(randn(3), [Time(0), Time(1), Time(2)])
    @test_throws ArgumentError regularize(tt)
    # A mixed-unit Period lookup has no single resolution to count in.
    mixed = Timeseries(randn(3), Dates.Period[Day(1), Hour(2), Hour(5)])
    @test_throws ArgumentError regularize(mixed)
end
