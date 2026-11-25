@testitem "ToolsArrays" tags = [:fast] begin
    import DimensionalData: span, DimArrayInterface
    import Interfaces

    Interfaces.test(
        DimArrayInterface, ToolsArray,
        [ToolsArray(randn(10, 10), (X(1:10), Y(1:10)))]
    )

    x = ToolsArray(randn(10), (𝑡(1:10),))
    @test x isa ToolsArray
    @test !(x isa DimensionalData.DimArray)
    @test x isa DimensionalData.AbstractDimArray

    using DimensionalData
    import DimensionalData: ForwardOrdered, Regular, Points, Sampled, Metadata, order,
        sampling, layerdims, index, locus, Intervals, intervalbounds,
        BasicDimArray
    a = [1 2; 3 4]
    a2 = [
        1 2 3 4
        3 4 5 6
        4 5 6 7
    ]
    xmeta = Metadata(:meta => "X")
    ymeta = Metadata(:meta => "Y")
    tmeta = Metadata(:meta => "T")
    ameta = Metadata(:meta => "da")
    dimz = (
        X(Sampled(143.0:2.0:145.0; order = ForwardOrdered(), metadata = xmeta)),
        Y(Sampled(-38.0:2.0:-36.0; order = ForwardOrdered(), metadata = ymeta)),
    )
    dimz2 = (Dim{:row}(10:10:30), Dim{:column}(-20:10:10))

    refdimz = (𝑡(1:1; metadata = tmeta),)
    da = @test_nowarn ToolsArray(a, dimz; refdims = refdimz, name = :test, metadata = ameta)
    val(dims(da, 1)) |> typeof
    da2 = ToolsArray(a2, dimz2; refdims = refdimz, name = :test2)
    lx = Sampled(143.0:2.0:145.0, ForwardOrdered(), Regular(2.0), Points(), xmeta)
    ly = Sampled(-38.0:2.0:-36.0, ForwardOrdered(), Regular(2.0), Points(), ymeta)
    db = DimArray(da)
    @test db isa DimArray
    @test dims(da) == dims(db)
    @test dims(db, X) == dims(da, X)
    @test refdims(db) == refdims(da)
    @test name(db) == name(da)
    @test metadata(db) == metadata(da)
    @test lookup(db) == lookup(da)
    @test order(db) == order(da)
    @test sampling(db) == sampling(da)
    @test span(db) == span(da)
    @test locus(db) == locus(da)
    @test bounds(db) == bounds(da)
    @test layerdims(db) == layerdims(da)
    # @test DimensionalData.index(db, Y) == DimensionalData.index(da, Y)
    da_intervals = set(da, X => Intervals, Y => Intervals)
    db_intervals = set(db, X => Intervals, Y => Intervals)
    @test intervalbounds(da_intervals) == intervalbounds(db_intervals)

    # * AbstractBasicDimArray constructor
    bmeta = Metadata(:meta => "B")
    bda = DimArray(randn(4), (𝑡(1:4),); name = :basic, metadata = bmeta)
    wrapped = @test_nowarn ToolsArray(bda; name = :wrapped_basic, metadata = bmeta)
    @test wrapped isa ToolsArray
    @test dims(wrapped) == dims(bda)
    @test parent(wrapped) == parent(bda)
    @test metadata(wrapped) == bmeta
    @test name(wrapped) == :wrapped_basic

    # * Function constructor
    f = sin
    y = @test_nowarn ToolsArray(f, 𝑡(1:10))
    @test lookup(y, 𝑡) == 1:10
    @test parent(y) == f.(1:10)

    f = +
    y = @test_nowarn ToolsArray(f, 𝑡(1:10), X(1:10))
    @test parent(y) == map(Iterators.product(1:10, 1:10)) do args
        f(args...)
    end

    # * Bad practice
    x = ToolsArray(randn(10), (𝑡(1:10),))
    @test parent(x) isa Vector
end

@testitem "Printing" tags = [:fast] begin
    # * Vector
    x = ToolsArray(randn(10), (𝑡(1:10),))
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * Matrix
    x = ToolsArray(randn(10, 10), (𝑡(1:10), 𝑥(1:10)))
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * Array
    x = ToolsArray(randn(2, 2, 2), (𝑡(1:2), 𝑥(1:2), 𝑦(1:2)))
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * 2D Array of arrays.
    x = ToolsArray(fill(randn(3, 2), 10, 10), (𝑡(1:10), 𝑥(1:10)))
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * 3D Array of arrays.
    x = ToolsArray(fill(randn(2, 2), 10, 10, 10), (𝑡(1:10), 𝑥(1:10), 𝑦(1:10)))
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * 4D Array of arrays.
    x = ToolsArray(fill(randn(2, 2), 3, 3), (𝑡(1:3), 𝑥(1:3)))
    x = ToolsArray(
        fill(x, 2, 1, 4, 1),
        (𝑡(1:2), 𝑥(1:1), 𝑦(1:4), 𝑧(1:1))
    )
    @test_nowarn show(x)
    @test_nowarn display(x)

    # * Spike train
    x = ToolsArray(rand(Bool, 5), (𝑡(1:5),))
    @test_nowarn show(x)
    @test_nowarn display(x)
end

@testitem "TimeseriesBase.jl" tags = [:fast] begin
    ts = 1:100
    x = @test_nowarn Timeseries(randn(100), ts)
    @test x isa AbstractTimeseries
    @test x isa RegularTimeseries
    @test x isa UnivariateTimeseries

    @test step(x) == step(ts)
    @test samplingrate(x) == 1 / step(ts)
    @test times(x) == ts
    @test duration(x) == -first(-(extrema(ts)...))
    @test Interval(x) == first(extrema(ts)) .. last(extrema(ts))
    @test x[𝑡(1 .. 10)] == x[1:10]
    @test all(x[𝑡(At(1:10))] .== x[1:10])
end

@testitem "Dim queries" tags = [:fast] begin
    ts = 1:100
    cs = 1:10
    x = Timeseries(randn(100, 10), ts, Dim{:channel}(cs))
    @test x isa ToolsArray
    @test dims(x, 1) isa ToolsDimension
    @test !(dims(x, 2) isa ToolsDimension)
    @test dims(x) == (𝑡(ts), Dim{:channel}(cs))
    @test dims(x, 1) == 𝑡(ts)
    @test dims(x, 𝑡) == 𝑡(ts)
    @test dims(x, Dim{:channel}) == Dim{:channel}(cs)
    @test dims(x, :channel) == Dim{:channel}(cs)

    DimensionalData.@dim U ToolsDim "U"
    @test U <: ToolsDimension

    x = ToolsArray(randn(10), (𝑥(1:10),))
    @test all(x[At(dims(x, 1))] .== x)
    @test lookup(x[At(dims(x, 1))]) != lookup(x) # One is Regular, one is Irregular
    @test all(lookup(x[At(dims(x, 1))]) .== lookup(x[At(1:10)])) # But same elements
end

@testitem "Multivariate time series" tags = [:fast] begin
    ts = 1:100
    x = @test_nowarn Timeseries(randn(100, 5), ts, 1:5)
    @test x isa AbstractTimeseries
    @test x isa RegularTimeseries
    @test x isa MultivariateTimeseries

    @test step(x) == step(ts)
    @test samplingrate(x) == 1 / step(ts)
    @test times(x) == ts
    @test duration(x) == -first(-(extrema(ts)...))
    @test Interval(x) == first(extrema(ts)) .. last(extrema(ts))
    @test x[𝑡(1 .. 10)] == x[1:10, :]
end

@testitem "Multidimensional time series" tags = [:fast] begin
    x = @test_nowarn Timeseries(randn(100, 10), 𝑡(1:100), X(1:10))
    @test x isa AbstractTimeseries
    @test x isa RegularTimeseries
    @test x isa MultidimensionalTimeseries

    x = @test_nowarn Timeseries(randn(100, 10, 10), 𝑡(1:100), X(1:10), Y(1:10))
    @test x isa AbstractTimeseries
    @test x isa RegularTimeseries
    @test x isa MultidimensionalTimeseries
    @test_nowarn x[𝑡(Near(4:10))]

    x = @test_nowarn Timeseries(randn(100, 10, 10), 𝑡(1:100), X(randn(10) |> sort), Y(1:10))
    @test x isa AbstractTimeseries
    @test x isa RegularTimeseries
    @test !(x isa MultidimensionalTimeseries)

    x = @test_nowarn Timeseries(randn(100), 𝑡(sort(randn(100))))
    @test x isa AbstractTimeseries
    @test !(x isa RegularTimeseries)
    @test !(x isa MultidimensionalTimeseries)
end

@testitem "Spatial dimensions" tags = [:fast] begin
    import DimensionalData as DD
    # 𝑥, 𝑦, 𝑧 must map to the matching DimensionalData spatial dimensions and
    # carry the matching labels (regression: 𝑦/𝑧 were swapped).
    @test 𝑥 <: DD.XDim
    @test 𝑦 <: DD.YDim
    @test 𝑧 <: DD.ZDim
    @test DD.label(𝑥(1:2)) == "x"
    @test DD.label(𝑦(1:2)) == "y"
    @test DD.label(𝑧(1:2)) == "z"

    # Every in-scope dimension (including the (log-)frequency dims) is a
    # ToolsDimension, so it dispatches to the ToolsArray constructor.
    for D in (𝑡, 𝑥, 𝑦, 𝑧, 𝑓, Log𝑓, Log10𝑓, Var, Obs)
        @test D <: ToolsDimension
    end
end

@testitem "ToolsArray{T} converts eltype and keeps everything else" tags = [:fast] begin
    A = ToolsArray(
        randn(4), (𝑡(1:4),); name = :orig, metadata = Dict(:k => 1),
        refdims = (Var(:a),)
    )
    B = ToolsArray{Float32}(A)
    @test eltype(B) == Float32
    @test parent(B) ≈ Float32.(parent(A))
    @test name(B) == :orig
    @test metadata(B) == Dict(:k => 1)
    @test refdims(B) == refdims(A)
    @test dims(B) == dims(A)
    @test name(ToolsArray{Float32}(A; name = :new)) == :new # kwargs must be honoured
    @test name(ToolsArray{Float64}(A; name = :same)) == :same
end

@testitem "relabel" tags = [:fast] begin
    import DimensionalData: order, ForwardOrdered, ReverseOrdered

    # * A reverse-ordered lookup relabelled with labels that sort forwards
    A = ToolsArray([1, 2, 3], (X([30, 20, 10]),); name = :a, metadata = Dict(:m => 1))
    B = relabel(A, X => [:a, :b, :c])
    @test B isa ToolsArray
    @test parent(B) == parent(A)
    @test lookup(B, X) == [:a, :b, :c]
    @test order(lookup(B, X)) == ForwardOrdered()
    @test B[X = At(:a)] == 1
    @test DimensionalData.name(B) == DimensionalData.name(A)
    @test DimensionalData.metadata(B) == DimensionalData.metadata(A)

    # * An unordered lookup relabelled with sorted labels
    C = ToolsArray([1, 2, 3], (X([:c, :a, :b]),))
    @test parent(relabel(C, X => [:p, :q, :r])) == [1, 2, 3]
    @test relabel(C, X => [:p, :q, :r])[X = At(:p)] == 1

    # * Negating an axis reverses its order but not the data
    D = ToolsArray([1, 2, 3], (X([10, 20, 30]),))
    E = relabel(D, X => -lookup(D, X))
    @test parent(E) == [1, 2, 3]
    @test E[X = At(-10)] == 1
    @test order(lookup(E, X)) == ReverseOrdered()

    # * A different dimension type
    F = relabel(A, X => Y([1.0, 2.0, 3.0]))
    @test dims(F, 1) isa Y
    @test parent(F) == parent(A)
    @test F[Y = Near(1.1)] == 1

    # * A bare dimension replaces the one of the same type; others are kept
    G = ToolsArray(reshape(1:6, 3, 2), (X([30, 20, 10]), Y([:u, :v])))
    H = relabel(G, X([:a, :b, :c]))
    @test parent(H) == parent(G)
    @test lookup(H, Y) == lookup(G, Y)
    @test H[X = At(:a), Y = At(:v)] == 4
    K = relabel(G, X => 1:3, :Y => [:v, :u])
    @test parent(K) == parent(G)
    @test K[X = At(1), Y = At(:v)] == 1

    # * Time series keep their type
    x = Timeseries(randn(10), 1:10)
    y = relabel(x, 𝑡 => 0.1:0.1:1.0)
    @test y isa AbstractTimeseries
    @test parent(y) == parent(x)
    @test times(y) == 0.1:0.1:1.0

    @test_throws DimensionMismatch relabel(A, X => [:a, :b])
    @test_throws ArgumentError relabel(A, X => [:a, :b, :c], X => [:d, :e, :f])
    @test_logs (:warn, r"more than one Y") relabel(G, X => Y(1:3))
    @test_logs (:warn, r"more than one Y") relabel(G, X => Dim{:Y}(1:3))
    S = @test_logs relabel(G, X => Y(1:3), Y => X([:u, :v])) # swapping does not warn
    @test parent(S) == parent(G)
    @test S[Y = At(1), X = At(:v)] == 4
    C = ToolsArray(reshape(1:9, 3, 3), (Dim{:neuron}(1:3), Dim{:neuron}(1:3)))
    R = @test_logs relabel(C, :neuron => [:a, :b, :c]) # existing duplicates do not warn
    @test lookup(R, 1) == [:a, :b, :c]
end
