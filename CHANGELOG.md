# Changelog

All notable changes to TimeseriesBase.jl are documented here.

## [v0.3.0]

Breaking. `Dates` time indices are now supported across windowing, grid repair and IO.

### Breaking
- `delayembed`: the delay spacing is now `τ * step(x)`, not `τ * p * step(x)`. The skip
  `p` sits between embedded vectors, not within one, so the `:delay` lookup changes for
  any call with `p != 1`.
- `regularize`/`rectify`: `atol` defaults to `1e-6 * step` (previously a `~1e3 * eps`
  floor) and is converted to the lookup's own units, so `100u"ms"` and `0.1u"s"` mean the
  same thing on a lookup in seconds. The same input can yield a different grid.
- `stitch` preserves an instant (`DateTime`, `Date`) time index, continuing the calendar
  from the first series' own start. It previously returned elapsed `Period`s from an
  implicit origin, silently dropping the calendar and with it `groupby`, `months` and the
  instant selectors. Numeric and `Period` indices are unchanged.
- `rectify`, `rectifytime` and `matchdim` reject a `Dates` lookup with an `ArgumentError`
  pointing at `regularize`. Their tolerance model (significant figures of a decimal step)
  has no meaning on integer counts.

### Added
- `Dates` support for `buffer`, `window`, `delayembed`, `coarsegrain` and `regularize`,
  over `DateTime`, `Date` and `Period` time indices.
- `regularize` on a `Dates` lookup fits the grid on exact integer offsets and accepts
  `atol` as a `Period`. Its default tolerance is exact, since a `Dates` lookup carries no
  float jitter, and a fitted step below the lookup's resolution is rejected.
- TSV IO round trips a `Dates` time column: instants travel as ISO 8601 and periods as
  bare counts, with the type recorded in the header.
- A JLD2 extension. `loadtoolsarray` and `toolsarray_typemap` reconstruct stored arrays
  whose custom `@dim` types are absent, degrading to a generic `Dim{name}` carrying the
  same lookup rather than losing the whole array.
- Docstrings for the exported dimensions (`𝑡`, `𝑥`, `𝑦`, `𝑧`, `Var`, `Obs`, `𝑓`, `Log𝑓`,
  `Log10𝑓`).
- Test coverage for `Dates` across windowing, grid repair, `stitch`, TSV and JLD2; the IO
  paths previously had none.

### Changed
- `freqs`, `FreqIndex` and `TimeFreqIndex` accept any `FrequencyDim`, not only `𝑓`.
- `samplingrate` and `nyquist` return a `Unitful` rate for a `Dates` index (a `Day(1)`
  step gives `1/86400 s^-1`) and throw for a calendar period (`Year`, `Month`,
  `Quarter`), which has no fixed length; use `samplingperiod`.
- `regularize(...; zero = true)` on a `Dates` lookup returns a uniform-unit `Period`
  range: `zero = true` means elapsed time, which is a duration, not an instant.
- `Dates.Time` and mixed-unit `Period` lookups are rejected with a clear message rather
  than failing deep in the arithmetic.

### Fixed
- `ℬ!`/`ℒ!` no longer alias their argument in place (`circshift!(x, n)` on itself is
  invalid).
- The `ToolsArray{T}` conversion constructor preserves dims, name and metadata.
- `delayembed` on a `Period`-stepped series: the delay range used a literal `0` as its
  endpoint, where a `Period` is needed.
- TSV load no longer returns `Vector{Any}` data when the time column holds strings.
- Selectors accept an unformatted dimension.

## [v0.2.1]

### Added
- `regularize`: new method for resampling irregular time series onto a regular grid.
- `nyquist` exported from `Utils`.
- Docstrings for `ℬ`/`ℬ!` (backshift/lag operators).
- Expanded test coverage for `Utils`, `IO`, and `ToolsArrays`.

### Changed
- SciMLBase v3 compatibility.
- CI now runs on schedule and supports manual dispatch.
- `interlace` narrowed from `AbstractTimeseries` to `UnivariateTimeseries`; multidimensional inputs previously produced silently incorrect output.
- `loadmultidimensionaltimeseries` (TSV) now throws `ArgumentError` instead of silently returning `nothing`.
- Code formatted with Runic.

## [v0.2.0] — 2026-03-05

Breaking release for DimensionalData.jl v0.30 and Julia v1.12.

### Added
- `Dropdims` callable struct: wraps a function and automatically drops singleton dimensions from its result.
- `SciMLBaseExt`: ODE solutions with duplicate timesteps (e.g. from `SavingCallback`) are handled correctly; callback values are split from the main series and stored in solution metadata under `:callback_values`.
- `SpikeTrain`/`SpikeTimes` types re-added after earlier removal.

### Changed
- **Breaking:** TSV time-column header changed from `"time"` to `"𝑡"` to match the internal dimension name.
- Julia v1.12 compatibility fixes throughout (`IO.jl`, imports).
- Removed deprecated `index` call from DimensionalData.

## [v0.1.4] — 2025-09-17

### Added
- Logarithmic frequency dimension types (`LogFrequency` hierarchy) with full export.

## [v0.1.3] — 2025-09-07

### Added
- `SciMLBaseExt` extension for DifferentialEquations.jl integration.

## [v0.1.2] — 2025-09-01

### Fixed
- Export corrections and type deprecation warnings.

## [v0.1.1] — 2025-08-25

### Fixed
- `SpikeTrain` constructor.

## [v0.1.0] — 2025-08-21

Initial release. Core timeseries types and utilities extracted from TimeseriesTools.jl, including `Timeseries`, `SpikeTrain`, `IO`, and basic rectification.
