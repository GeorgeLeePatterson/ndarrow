# Capability Matrix

## Status Legend

| Symbol | Meaning       |
|--------|---------------|
| Done   | Implemented, tested, documented |
| WIP    | In progress   |
| --     | Not started   |

## Inbound: Arrow -> ndarray (Zero-Copy Views)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| PrimitiveArray<f32> -> ArrayView1   | Done   | Implemented via `AsNdarray`                  |
| PrimitiveArray<f64> -> ArrayView1   | Done   | Implemented via `AsNdarray`                  |
| FixedSizeList<f32> -> ArrayView2    | Done   | Implemented (`validated`/`unchecked`/`masked`) |
| FixedSizeList<f64> -> ArrayView2    | Done   | Implemented (`validated`/`unchecked`/`masked`) |
| FixedShapeTensor -> ArrayViewD      | --     | Shape from extension metadata                |
| VariableShapeTensor -> per-row view | --     | Iterator over per-element ArrayViewD         |
| ndarrow.csr_matrix -> CsrView       | --     | Borrow offsets + indices + values            |
| Two-column sparse -> CsrView       | --     | Convenience API for separate columns         |

## Outbound: ndarray -> Arrow (Ownership Transfer)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Array1<f32> -> PrimitiveArray       | Done   | Implemented via `IntoArrow`                  |
| Array1<f64> -> PrimitiveArray       | Done   | Implemented via `IntoArrow`                  |
| Array2<f32> -> FixedSizeList        | Done   | Implemented via `IntoArrow`                  |
| Array2<f64> -> FixedSizeList        | Done   | Implemented via `IntoArrow`                  |
| ArrayD<T> -> FixedShapeTensor       | --     | into_raw_vec + shape metadata                |
| CsrMatrix-like -> ndarrow.csr_matrix | --     | Transfer row_ptrs, indices, values           |

## Null Handling

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| Validated (null_count check)        | Done   | Default tier, returns Result                 |
| Unchecked (caller guarantees)       | Done   | Zero-cost, unsafe                            |
| Masked (view + validity bitmap)     | Done   | Returns tuple, zero allocation               |
| fill_nulls(zero)                    | --     | Allocating helper                            |
| fill_nulls(mean)                    | --     | Allocating helper, requires float type       |
| compact_non_null                    | --     | Allocating helper, removes null rows         |

## Helpers (Explicit Allocation Points)

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| cast f32 -> f64                     | --     | Element-wise widening                        |
| cast f64 -> f32                     | --     | Element-wise ndarrowing                       |
| densify sparse -> dense             | --     | Sparse to FixedSizeList                      |
| reshape PrimitiveArray -> 2D view   | --     | Zero-copy if dimensions align                |
| reshape PrimitiveArray -> ND view   | --     | Zero-copy if dimensions align                |
| to_standard_layout                  | --     | No-op if already C-contiguous                |

## Extension Types

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| arrow.fixed_shape_tensor support    | --     | Read/write canonical tensor extension        |
| arrow.variable_shape_tensor support | --     | Read/write canonical variable tensor         |
| ndarrow.csr_matrix definition        | --     | Custom sparse extension type                 |
| Extension type registration         | --     | Register handlers for deserialization        |

## Element Type Support

| Type       | Inbound | Outbound | Notes                               |
|------------|---------|----------|-------------------------------------|
| f32        | Done    | Done     | First-class, primary                |
| f64        | Done    | Done     | First-class, primary                |
| Complex32  | --      | --       | Future, pending trait bounds        |
| Complex64  | --      | --       | Future, pending trait bounds        |
| i32        | --      | --       | Future, for index arrays            |
| i64        | --      | --       | Future, for index arrays            |
| u32        | --      | --       | Future, for sparse indices          |
| u64        | --      | --       | Future                              |

## Core Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| NdarrowElement trait                 | Done   | Type bridge: Arrow <-> ndarray               |
| AsNdarray trait                     | Done   | Inbound conversion contract                  |
| IntoArrow trait                     | Done   | Outbound conversion contract                 |
| NdarrowError enum                    | Done   | Error taxonomy                               |
| Prelude module                      | --     | Convenience re-exports                       |

## Quality Infrastructure

| Capability                          | Status | Notes                                        |
|-------------------------------------|--------|----------------------------------------------|
| justfile                            | Done   | Build/test/lint/coverage/release commands    |
| CI pipeline                         | Done   | GitHub Actions                               |
| Coverage >= 90%                     | Done   | Gate configured in just/CI                   |
| Benchmarks                          | Done   | Public API conversion benchmark suites       |
| Property tests                      | --     | Round-trip Arrow -> ndarray -> Arrow          |
