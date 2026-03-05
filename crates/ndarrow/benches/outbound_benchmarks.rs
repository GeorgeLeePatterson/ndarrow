//! Benchmarks for ndarray -> Arrow (outbound) conversions.

use criterion::{Criterion, criterion_group, criterion_main};

fn outbound_benchmarks(_c: &mut Criterion) {
    // TODO: Add benchmarks for Array1 -> PrimitiveArray and Array2 -> FixedSizeListArray.
}

criterion_group!(benches, outbound_benchmarks);
criterion_main!(benches);
