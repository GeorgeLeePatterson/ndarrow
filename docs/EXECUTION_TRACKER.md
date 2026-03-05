# Execution Tracker

Last updated: 2026-03-05

## Purpose

Operational state tracker for ndarrow implementation progress.

## Resume Protocol

1. Read `docs/README.md`.
2. Read `docs/DECISIONS.md` (locked constraints).
3. Read this file for current done/next state.
4. Read `docs/architecture.md` for structural work.

## Done

| ID    | Description | Phase |
|-------|-------------|-------|
| D-001 | Planning docs and architectural decisions completed | Pre |
| D-002 | AGENTS/quality-gate baseline established | Pre |
| I-001 | Workspace dependencies and lint config wired | 1 |
| I-002 | `NdarrowError` implemented | 1 |
| I-003 | `NdarrowElement` implemented (`f32`/`f64`) | 1 |
| I-004 | `AsNdarray` trait implemented for primitive inbound path | 2 |
| I-005 | `FixedSizeList -> ArrayView2` inbound conversions implemented | 2 |
| I-006 | `IntoArrow` trait implemented for `Array1`/`Array2` outbound path | 3 |
| I-007 | Explicit null handling tiers implemented (`validated`/`unchecked`/`masked`) | 4 |
| I-008 | Dense round-trip integration tests added | 9 |
| I-009 | Zero-copy pointer verification tests added | 9 |
| I-010 | Null handling integration tests added | 9 |
| I-011 | CI pipeline and coverage gate added | 10 |
| I-012 | Public API benchmark suites added for inbound/outbound conversions | 9 |
| I-013 | Benchmark smoke command aligned to quick smoke convention | 9 |
| I-014 | nabled interop tracker updated with verified NC-001/NC-002/NC-003 completion | 10 |

## Next (Priority Order)

| ID    | Description | Phase | Priority |
|-------|-------------|-------|----------|
| N-013 | `ndarrow.csr_matrix` extension type definition | 5 | P1 |
| N-014 | `CsrView` type definition | 5 | P1 |
| N-015 | Sparse inbound (extension type + two-column) | 5 | P1 |
| N-016 | Sparse outbound | 5 | P1 |
| N-017 | FixedShapeTensor inbound | 6 | P1 |
| N-018 | FixedShapeTensor outbound | 6 | P1 |
| N-019 | VariableShapeTensor inbound iterator | 7 | P1 |
| N-020 | VariableShapeTensor outbound | 7 | P1 |
| N-021 | `cast` helper | 8 | P2 |
| N-022 | `densify` helper | 8 | P2 |
| N-023 | `reshape` helper | 8 | P2 |
| N-024 | Property tests | 9 | P2 |
| N-025 | Benchmark baseline persistence/reporting refinements | 9 | P2 |
| N-026 | Allocation verification expansion for sparse/tensor paths | 9 | P2 |
| N-027 | Documentation polish for newly added capabilities | 10 | P2 |

## Needed (Open Questions)

| ID    | Description | Notes |
|-------|-------------|-------|
| K-004 | Verify FixedShapeTensor API details in current arrow-rs | Feature semantics and metadata shape contract |
| K-005 | Verify VariableShapeTensor feature/stability in arrow-rs | Extension/type availability and ergonomics |
| K-007 | Decide complex Arrow representation strategy for ndarrow | NC-004 equivalent, ndarrow-owned design |
| K-008 | Property-test strategy depth (`proptest` scenarios) | Sparse/tensor invariants and round-trip laws |
