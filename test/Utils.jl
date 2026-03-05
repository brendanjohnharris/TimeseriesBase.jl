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
    @test y == z

    z = ToolsArray([x, x], 𝑓(1:2))
    y = stack(z; dims = 1)
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
