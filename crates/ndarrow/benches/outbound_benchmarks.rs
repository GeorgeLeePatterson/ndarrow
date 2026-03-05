//! Benchmarks for ndarray -> Arrow (outbound) conversion APIs.

use std::hint::black_box;

use arrow_array::{Array, FixedSizeListArray, Float32Array, Float64Array};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::{Array1, Array2};
use ndarrow::IntoArrow;

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

fn make_array2_f32(rows: usize, cols: usize) -> Array2<f32> {
    let len = rows * cols;
    let data = values_f32(len);
    Array2::from_shape_vec((rows, cols), data).expect("shape must match data length")
}

fn make_array2_f64(rows: usize, cols: usize) -> Array2<f64> {
    let len = rows * cols;
    let data = values_f64(len);
    Array2::from_shape_vec((rows, cols), data).expect("shape must match data length")
}

fn bench_array1_outbound(c: &mut Criterion) {
    let mut group = c.benchmark_group("outbound_array1");

    for len in [256_usize, 4096, 65_536] {
        let f32_template = Array1::from_vec(values_f32(len));
        group.bench_with_input(
            BenchmarkId::new("into_arrow_f32", len),
            &f32_template,
            |bench, t| {
                bench.iter_batched(
                    || t.clone(),
                    |array| {
                        let converted: Float32Array =
                            array.into_arrow().expect("Array1<f32> conversion must succeed");
                        black_box(converted.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        let f64_template = Array1::from_vec(values_f64(len));
        group.bench_with_input(
            BenchmarkId::new("into_arrow_f64", len),
            &f64_template,
            |bench, t| {
                bench.iter_batched(
                    || t.clone(),
                    |array| {
                        let converted: Float64Array =
                            array.into_arrow().expect("Array1<f64> conversion must succeed");
                        black_box(converted.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_array2_outbound(c: &mut Criterion) {
    let mut group = c.benchmark_group("outbound_array2");

    for (rows, cols) in [(128_usize, 32_usize), (1024, 128)] {
        let id = format!("{rows}x{cols}");

        let f32_template = make_array2_f32(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("into_arrow_f32", &id),
            &f32_template,
            |bench, t| {
                bench.iter_batched(
                    || t.clone(),
                    |array| {
                        let converted: FixedSizeListArray =
                            array.into_arrow().expect("Array2<f32> conversion must succeed");
                        black_box(converted.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        let f64_template = make_array2_f64(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("into_arrow_f64", &id),
            &f64_template,
            |bench, t| {
                bench.iter_batched(
                    || t.clone(),
                    |array| {
                        let converted: FixedSizeListArray =
                            array.into_arrow().expect("Array2<f64> conversion must succeed");
                        black_box(converted.len());
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn outbound_benchmarks(c: &mut Criterion) {
    bench_array1_outbound(c);
    bench_array2_outbound(c);
}

criterion_group!(benches, outbound_benchmarks);
criterion_main!(benches);
