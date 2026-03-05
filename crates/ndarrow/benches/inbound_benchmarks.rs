//! Benchmarks for Arrow -> ndarray (inbound) conversion APIs.

use std::{hint::black_box, sync::Arc};

use arrow_array::{
    FixedSizeListArray, Float32Array, Float64Array,
    types::{Float32Type, Float64Type},
};
use arrow_schema::{DataType, Field};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarrow::{AsNdarray, fixed_size_list_as_array2};

fn values_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let bounded = u16::try_from(i % 1024).expect("value must be in u16 range");
            f32::from(bounded) * 0.001
        })
        .collect()
}

fn values_f64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let bounded = u16::try_from(i % 1024).expect("value must be in u16 range");
            f64::from(bounded) * 0.001
        })
        .collect()
}

fn make_fsl_f32(rows: usize, dim: i32) -> FixedSizeListArray {
    let dim_usize = usize::try_from(dim).expect("dim must be non-negative");
    let len = rows * dim_usize;
    let values = Float32Array::from(values_f32(len));
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    FixedSizeListArray::new(field, dim, Arc::new(values), None)
}

fn make_fsl_f64(rows: usize, dim: i32) -> FixedSizeListArray {
    let dim_usize = usize::try_from(dim).expect("dim must be non-negative");
    let len = rows * dim_usize;
    let values = Float64Array::from(values_f64(len));
    let field = Arc::new(Field::new("item", DataType::Float64, false));
    FixedSizeListArray::new(field, dim, Arc::new(values), None)
}

fn bench_primitive_inbound(c: &mut Criterion) {
    let mut group = c.benchmark_group("inbound_primitive");

    for len in [256_usize, 4096, 65_536] {
        let f32_array = Float32Array::from(values_f32(len));
        group.bench_with_input(
            BenchmarkId::new("as_ndarray_f32", len),
            &f32_array,
            |bench, array| {
                bench.iter(|| {
                    let view = array.as_ndarray().expect("array must not contain nulls");
                    black_box(view.len());
                });
            },
        );

        let f64_array = Float64Array::from(values_f64(len));
        group.bench_with_input(
            BenchmarkId::new("as_ndarray_f64", len),
            &f64_array,
            |bench, array| {
                bench.iter(|| {
                    let view = array.as_ndarray().expect("array must not contain nulls");
                    black_box(view.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_fixed_size_list_inbound(c: &mut Criterion) {
    let mut group = c.benchmark_group("inbound_fixed_size_list");

    for (rows, dim) in [(128_usize, 32_i32), (1024, 128)] {
        let id = format!("{rows}x{dim}");
        let f32_array = make_fsl_f32(rows, dim);
        group.bench_with_input(
            BenchmarkId::new("fixed_size_list_as_array2_f32", &id),
            &f32_array,
            |bench, array| {
                bench.iter(|| {
                    let view = fixed_size_list_as_array2::<Float32Type>(array)
                        .expect("fixed-size list must be valid and non-null");
                    black_box(view.dim());
                });
            },
        );

        let f64_array = make_fsl_f64(rows, dim);
        group.bench_with_input(
            BenchmarkId::new("fixed_size_list_as_array2_f64", &id),
            &f64_array,
            |bench, array| {
                bench.iter(|| {
                    let view = fixed_size_list_as_array2::<Float64Type>(array)
                        .expect("fixed-size list must be valid and non-null");
                    black_box(view.dim());
                });
            },
        );
    }

    group.finish();
}

fn inbound_benchmarks(c: &mut Criterion) {
    bench_primitive_inbound(c);
    bench_fixed_size_list_inbound(c);
}

criterion_group!(benches, inbound_benchmarks);
criterion_main!(benches);
