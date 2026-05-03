# Rename Package Design

## Background

The old top-level directory name represented a reproduction-only experiment. The project now contains a broader PM2.5 forecasting workflow with multiple models and comparison outputs, so the package name should describe the current scope.

## Decision

Rename the top-level Python package from the legacy reproduction package to `pm25_forecast`.

Update these surfaces:

- Python imports now use `pm25_forecast`.
- Module commands now use `python -m pm25_forecast.scripts...`.
- Default outputs now live under `pm25_forecast/outputs/...`.
- Tests and docs use the new package path.

## Compatibility

Do not migrate or delete existing experiment artifacts. New commands write to the new package output directory. Do not keep a compatibility alias for the old package name, because two package names would make future commands and docs ambiguous.

## Verification

- Unit tests pass.
- Main CLI `--help` commands load under the new package name.
- Data preparation can generate the default `720h -> 72h` bundle.
