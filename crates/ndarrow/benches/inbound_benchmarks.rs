//! Benchmarks for Arrow -> ndarray (inbound) conversions.

use criterion::{Criterion, criterion_group, criterion_main};

fn inbound_benchmarks(_c: &mut Criterion) {
    // TODO: Add benchmarks for PrimitiveArray -> ArrayView1 and FixedSizeList -> ArrayView2.
}

criterion_group!(benches, inbound_benchmarks);
criterion_main!(benches);
