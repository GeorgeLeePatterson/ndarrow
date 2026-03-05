//! Integration tests verifying round-trip correctness.
//!
//! Every value that goes Arrow → ndarray → Arrow (or vice versa) must come back identical.
//! These tests exercise the public API exactly as a user would.

use approx::assert_abs_diff_eq;
use arrow_array::{
    Array, Float32Array, Float64Array,
    types::{Float32Type, Float64Type},
};
use ndarray::{Array1, Array2, array};
use ndarrow::{AsNdarray, IntoArrow};
use num_traits::FromPrimitive;

// ─── Scalar round-trips ───

#[test]
fn f64_arrow_to_ndarray_to_arrow() {
    let original = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Arrow → ndarray view
    let view = original.as_ndarray().unwrap();
    assert_eq!(view.len(), 5);

    // Copy view into owned array (simulating a computation result)
    let owned = view.to_owned();

    // ndarray → Arrow
    let result: Float64Array = owned.into_arrow().unwrap();

    assert_eq!(result.len(), original.len());
    for i in 0..result.len() {
        assert_abs_diff_eq!(result.value(i), original.value(i));
    }
}

#[test]
fn f32_arrow_to_ndarray_to_arrow() {
    let original = Float32Array::from(vec![1.0_f32, 2.0, 3.0]);

    let view = original.as_ndarray().unwrap();
    let owned = view.to_owned();
    let result: Float32Array = owned.into_arrow().unwrap();

    assert_eq!(result.len(), original.len());
    for i in 0..result.len() {
        assert_abs_diff_eq!(result.value(i), original.value(i));
    }
}

#[test]
fn f64_ndarray_to_arrow_to_ndarray() {
    let original = Array1::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0]);
    let expected = original.clone();

    // ndarray → Arrow
    let arrow: Float64Array = original.into_arrow().unwrap();

    // Arrow → ndarray view
    let view = arrow.as_ndarray().unwrap();

    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

#[test]
fn f32_ndarray_to_arrow_to_ndarray() {
    let original = Array1::from_vec(vec![1.5_f32, 2.5, 3.5]);
    let expected = original.clone();

    let arrow: Float32Array = original.into_arrow().unwrap();
    let view = arrow.as_ndarray().unwrap();

    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

// ─── Matrix round-trips ───

#[test]
fn f64_matrix_arrow_to_ndarray_to_arrow() {
    use std::sync::Arc;

    use arrow_schema::Field;

    // Build a FixedSizeListArray: 3 rows of 4-dimensional vectors
    let values = Float64Array::from(vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ]);
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float64, false));
    let fsl = arrow_array::FixedSizeListArray::new(field, 4, Arc::new(values), None);

    // Arrow → ndarray
    let view = ndarrow::fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();
    assert_eq!(view.dim(), (3, 4));

    // Simulate computation: double all values
    let doubled = &view * 2.0;

    // ndarray → Arrow
    let result = doubled.into_arrow().unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.value_length(), 4);

    // Verify values
    let inner = result.values().as_any().downcast_ref::<Float64Array>().unwrap();
    let result_values: Vec<f64> = inner.values().iter().copied().collect();
    let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];
    assert_eq!(result_values.len(), expected.len());
    for (actual, expected) in result_values.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

#[test]
fn f64_matrix_ndarray_to_arrow_to_ndarray() {
    let original = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let expected = original.clone();

    let fsl = original.into_arrow().unwrap();
    let view = ndarrow::fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();

    assert_eq!(view.dim(), expected.dim());
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

#[test]
fn f32_matrix_ndarray_to_arrow_to_ndarray() {
    let original = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let expected = original.clone();

    let fsl = original.into_arrow().unwrap();
    let view = ndarrow::fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();

    assert_eq!(view.dim(), expected.dim());
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

// ─── Edge cases ───

#[test]
fn empty_scalar_roundtrip() {
    let arr = Float64Array::from(Vec::<f64>::new());
    let view = arr.as_ndarray().unwrap();
    assert_eq!(view.len(), 0);

    let owned = view.to_owned();
    let result: Float64Array = owned.into_arrow().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn single_element_roundtrip() {
    let arr = Float64Array::from(vec![42.0]);
    let view = arr.as_ndarray().unwrap();
    let result: Float64Array = view.to_owned().into_arrow().unwrap();
    assert_abs_diff_eq!(result.value(0), 42.0);
}

#[test]
fn large_array_roundtrip() {
    let data: Vec<f64> = (0..10_000).map(f64::from).collect();
    let arr = Float64Array::from(data.clone());

    let view = arr.as_ndarray().unwrap();
    let result: Float64Array = view.to_owned().into_arrow().unwrap();

    assert_eq!(result.len(), 10_000);
    for (i, expected) in data.iter().enumerate() {
        assert_abs_diff_eq!(result.value(i), *expected);
    }
}

#[test]
fn large_matrix_roundtrip() {
    // 1000 rows of 128-dimensional vectors (typical embedding size)
    let data: Vec<f32> = (0_u32..128_000_u32)
        .map(|i| f32::from_u32(i).expect("u32 to f32 conversion should succeed for test range"))
        .collect();
    let matrix = Array2::from_shape_vec((1000, 128), data).unwrap();
    let expected = matrix.clone();

    let fsl = matrix.into_arrow().unwrap();
    assert_eq!(fsl.len(), 1000);
    assert_eq!(fsl.value_length(), 128);

    let view = ndarrow::fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();
    for (actual, expected) in view.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*actual, *expected);
    }
}

// ─── Computation pipeline ───

#[test]
fn realistic_computation_pipeline() {
    // Simulate: receive vectors from Arrow, normalize, return to Arrow.
    let raw_data: Vec<f32> = vec![
        3.0, 4.0, // row 0: norm = 5
        0.0, 1.0, // row 1: norm = 1
        1.0, 0.0, // row 2: norm = 1
    ];
    let values = Float32Array::from(raw_data);
    let field = std::sync::Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        false,
    ));
    let fsl = arrow_array::FixedSizeListArray::new(field, 2, std::sync::Arc::new(values), None);

    // Arrow → ndarray
    let view = ndarrow::fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();

    // Normalize each row (L2)
    let mut result = view.to_owned();
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }

    // ndarray → Arrow
    let output = result.into_arrow().unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output.value_length(), 2);

    // Verify normalized values
    let inner = output.values().as_any().downcast_ref::<Float32Array>().unwrap();
    let vals: Vec<f32> = inner.values().iter().copied().collect();

    // Row 0: [3/5, 4/5] = [0.6, 0.8]
    assert!((vals[0] - 0.6).abs() < 1e-6);
    assert!((vals[1] - 0.8).abs() < 1e-6);
    // Row 1: [0, 1]
    assert!((vals[2] - 0.0).abs() < 1e-6);
    assert!((vals[3] - 1.0).abs() < 1e-6);
    // Row 2: [1, 0]
    assert!((vals[4] - 1.0).abs() < 1e-6);
    assert!((vals[5] - 0.0).abs() < 1e-6);
}
