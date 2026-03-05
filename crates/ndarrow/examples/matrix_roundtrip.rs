//! Matrix (2D array) round-trip conversion.
//!
//! Demonstrates the zero-copy bridge for 2D data:
//! - Arrow `FixedSizeListArray` to ndarray `ArrayView2`
//! - ndarray `Array2` to Arrow `FixedSizeListArray`
//!
//! This is the pattern used for vector embeddings: each row is one
//! fixed-dimensional vector, stored as Arrow's `FixedSizeList<Float32>(D)`.
//!
//! Run with: `cargo run -p ndarrow --example matrix_roundtrip`

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, types::Float32Type};
use arrow_schema::Field;
use ndarray::Array2;
use ndarrow::IntoArrow;

fn main() {
    println!("=== ndarrow: Matrix Round-Trip Example ===\n");

    // ─── Create embedding-like data in ndarray ───
    println!("1. Create a matrix of 4 vectors, each 3-dimensional");
    let embeddings = Array2::from_shape_vec(
        (4, 3),
        vec![
            0.1, 0.2, 0.3, // vector 0
            0.4, 0.5, 0.6, // vector 1
            0.7, 0.8, 0.9, // vector 2
            1.0, 1.1, 1.2, // vector 3
        ],
    )
    .unwrap();
    println!("   ndarray shape: {:?}", embeddings.dim());
    println!("   {embeddings}\n");

    // ─── ndarray -> Arrow FixedSizeList ───
    println!("2. Convert to Arrow FixedSizeListArray (zero-copy)");
    let fsl: FixedSizeListArray = embeddings.into_arrow().unwrap();
    println!("   Arrow rows: {}", fsl.len());
    println!("   List size (dim): {}", fsl.value_length());
    println!("   Null count: {}\n", fsl.null_count());

    // ─── Arrow -> ndarray view ───
    println!("3. Convert back to ndarray view (zero-copy)");
    let view = ndarrow::fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();
    println!("   ndarray shape: {:?}", view.dim());
    println!("   {view}\n");

    // ─── Simulate receiving data from Arrow (e.g., DataFusion query result) ───
    println!("4. Simulate: receive vectors from Arrow, normalize, return to Arrow");

    let raw_values = Float32Array::from(vec![
        3.0, 4.0, 0.0, // vector 0: norm = 5
        0.0, 0.0, 1.0, // vector 1: norm = 1
    ]);
    let field = Arc::new(Field::new("item", arrow_schema::DataType::Float32, false));
    let input = FixedSizeListArray::new(field, 3, Arc::new(raw_values), None);

    // Arrow → ndarray
    let matrix = ndarrow::fixed_size_list_as_array2::<Float32Type>(&input).unwrap();
    println!("   Input matrix:");
    println!("   {matrix}");

    // Normalize each row (L2 normalization)
    let mut result = matrix.to_owned();
    for mut row in result.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }

    println!("   After L2 normalization:");
    println!("   {result}");

    // ndarray → Arrow
    let output: FixedSizeListArray = result.into_arrow().unwrap();
    println!("   Output Arrow rows: {}", output.len());
    println!("   Output list size: {}", output.value_length());
    println!("\n   Done! The only allocation was the normalization computation.");
}
