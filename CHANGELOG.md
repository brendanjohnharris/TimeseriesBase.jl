# Changelog

All notable changes to TimeseriesBase.jl are documented here.

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
