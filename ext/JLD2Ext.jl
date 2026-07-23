module JLD2Ext

import JLD2
using TimeseriesBase
using DimensionalData
const DD = DimensionalData
import TimeseriesBase: loadtoolsarray, toolsarray_typemap

# JLD2 bakes the concrete container type --- including every dimension type --- into
# the stored type. When a custom `@dim` type is absent on read, JLD2 resolves the
# read-as type to `UnknownType` and degrades the whole array to an opaque struct.
# The fix is load-side: force JLD2 to `Upgrade` the container to the bare
# `ToolsArray`/`DimArray`, reconstruct the fields, and rebuild via `name2dim`
# (which returns the specific dim type if present, else a generic `Dim{name}`) with
# the original lookup re-attached. `writeas` cannot do this --- it never bypasses the
# poisoned read-as type --- so this is deliberately read-only; writing is unchanged.

# Name of a dim, from either a real Dimension or an opaque JLD2-reconstructed struct.
function _recon_dim_name(d)
    d isa DD.Dimension && return DD.name(d)
    tname = string(typeof(d).parameters[1])            # e.g. "Main.M.Foo{…}" / "…Dim{:custom,…}"
    head = last(split(first(split(tname, ('{', ',', ' '))), '.'))
    if head == "Dim"
        inner = split(tname, '{'; limit = 2)[2]
        return Symbol(first(split(inner, (',', '}'))))  # the :custom in Dim{:custom}
    end
    return Symbol(head)
end

# The stored lookup (the whole `Lookup`, not bare values, so `format` does not re-infer). When the
# lookup's own type is absent here (a custom `Lookup`, or one whose values are a `ToolsArray`), JLD2
# hands back a reconstructed struct that `format` rejects; fall back to its raw values (any nested
# `ToolsArray` is already upgraded by the typemap) so `format` re-infers a plain `Lookup`.
function _recon_dim_val(d)
    d isa DD.Dimension && return DD.val(d)
    v = getproperty(d, :val)
    (v isa JLD2.ReconstructedStatic || v isa JLD2.ReconstructedMutable) &&
        return collect(getproperty(v, :data))
    return v
end

function _rebuild_dims(ds)
    out = map(ds) do d
        nm = _recon_dim_name(d)
        base = DD.name2dim(Val(nm))                    # specific type if loaded, else Dim{nm}()
        if !(d isa DD.Dimension) && base isa DD.Dim
            @warn "custom dimension :$nm is not defined in this workspace; \
                   reconstructing as Dim{:$nm} (lookup preserved)" maxlog = 16
        end
        DD.rebuild(base, _recon_dim_val(d))            # exact lookup re-attached
    end
    return Tuple(out)
end

function _rebuild(constructor, nt)
    # `Upgrade` types each reconstructed cell as the bare `ToolsArray` UnionAll, so a
    # container of ToolsArrays comes back with a widened (abstract) eltype. `map`
    # re-narrows to the tightest concrete eltype (so nested grids round-trip
    # type-identically) while preserving dimensionality (unlike broadcasting, which
    # collapses a 0-dim array to a scalar).
    data = map(identity, nt.data)
    return constructor(
        data, _rebuild_dims(nt.dims);
        refdims = _rebuild_dims(nt.refdims), name = nt.name, metadata = nt.metadata,
    )
end

# `Upgrade` sets the read-as type to the bare `ToolsArray`/`DimArray` UnionAll, so
# these methods dispatch even when a dim parameter is unresolvable. Foreign
# `metadata`/`name` values arrive behind `Any` and so cannot cascade.
JLD2.rconvert(::Type{<:AbstractToolsArray}, nt::NamedTuple) = _rebuild(ToolsArray, nt)
JLD2.rconvert(::Type{<:DimArray}, nt::NamedTuple) = _rebuild(DimArray, nt)

# Map any stored ToolsArray (incl. the legacy `TimeseriesTools.ToolsArray` path) and
# any plain DimArray to an `Upgrade`; everything else uses JLD2's default resolution.
function toolsarray_typemap(f, path::AbstractString, params)
    endswith(path, "ToolsArray") && return JLD2.Upgrade(ToolsArray)
    endswith(path, "DimArray") && return JLD2.Upgrade(DimArray)
    return JLD2.default_typemap(f, path, params)
end

loadtoolsarray(file::AbstractString, key) = JLD2.load(file, key; typemap = toolsarray_typemap)

end
