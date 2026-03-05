# Status — Current Snapshot

Last updated: 2026-03-05

## Summary

ndarrow is in the **foundation complete / dense bridge implemented** state.
Core traits, error model, dense inbound/outbound conversions, explicit null handling tiers, CI,
and release automation are implemented.

Sparse, tensor extension-type support, and helper APIs remain pending.

## Crate State

| Aspect                | Status |
|-----------------------|--------|
| Cargo.toml            | Workspace and crate dependencies configured (`arrow*`, `ndarray`, test/tooling deps). |
| src/lib.rs            | Public API module wiring and re-exports implemented. |
| Module tree           | `element`, `error`, `inbound`, `outbound` implemented. |
| Dependencies          | Added and pinned at workspace level. |
| Tests                 | Unit + integration tests for dense, null semantics, and zero-copy behavior. |
| CI                    | Implemented (`fmt`, `clippy`, feature checks, unit/integration tests, coverage, bench smoke). |
| Coverage              | Gate configured at 90% line coverage. |

## Implemented Capability Baseline

1. `NdarrowElement` trait with `f32`/`f64` support.
2. `NdarrowError` taxonomy.
3. `AsNdarray` for `PrimitiveArray<T>`.
4. `FixedSizeListArray -> ArrayView2<T>` conversion APIs (`validated`, `unchecked`, `masked`).
5. `IntoArrow` for `Array1<T>` and `Array2<T>`.
6. Integration tests for round-trip correctness and zero-copy pointer guarantees.
7. Benchmark harness with smoke-compatible public API conversion benchmarks.

## Dependencies on Upstream Changes

See `NABLED_CHANGES.md` for detail.

| Change | Crate | Status | Blocking? |
|--------|-------|--------|-----------|
| First-class `f32` support | nabled | Completed in nabled `main`; publish pending | No |
| `CsrMatrixView` with Arrow-native index types | nabled | Completed in nabled `main`; publish pending | No |
| View-accepting sparse ops | nabled | Completed in nabled `main`; publish pending | No |
| Complex Arrow representation assessment | nabled | Out of nabled scope (tracked on ndarrow side) | No |

## Constraints In Force

1. Zero-copy bridge semantics for view/ownership-transfer paths.
2. Vendor agnosticism (no producer/consumer coupling).
3. ndarray independence (no hard dependency on nabled).
4. Explicit null handling at call sites.
5. Quality gates: `fmt`, `clippy -D warnings -W clippy::pedantic`, coverage >= 90%.

## Next Milestone

**Phase 5+ backlog**:

1. Sparse extension and `CsrView` APIs.
2. Tensor extension-type inbound/outbound APIs.
3. Explicit helper APIs (`cast`, `densify`, `reshape`) and property-test expansion.
