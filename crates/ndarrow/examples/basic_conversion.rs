//! Basic Arrow <-> ndarray conversions.
//!
//! Shows the fundamental zero-copy bridge operations:
//! - Arrow `PrimitiveArray` to ndarray `ArrayView1`
//! - ndarray `Array1` to Arrow `PrimitiveArray`
//!
//! Run with: `cargo run -p ndarrow --example basic_conversion`

use arrow_array::{Array, Float64Array, PrimitiveArray, types::Float64Type};
use ndarray::Array1;
use ndarrow::{AsNdarray, IntoArrow};

fn main() {
    println!("=== ndarrow: Basic Conversion Example ===\n");

    // ─── Arrow to ndarray (zero-copy view) ───
    println!("1. Arrow -> ndarray (zero-copy view)");

    let arrow_array = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("   Arrow array: {:?}", arrow_array.values());

    let view = arrow_array.as_ndarray().unwrap();
    println!("   ndarray view: {view}");
    println!("   view[0] = {}", view[0]);
    println!("   view[4] = {}", view[4]);
    println!("   (No allocation occurred — the view borrows Arrow's buffer)\n");

    // ─── ndarray to Arrow (zero-copy ownership transfer) ───
    println!("2. ndarray -> Arrow (zero-copy ownership transfer)");

    let ndarray_result = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
    println!("   ndarray array: {ndarray_result}");

    let arrow_result: PrimitiveArray<Float64Type> = ndarray_result.into_arrow().unwrap();
    println!("   Arrow array: {:?}", arrow_result.values());
    println!("   length = {}", arrow_result.len());
    println!("   null_count = {}", arrow_result.null_count());
    println!("   (No allocation occurred — ownership was transferred)\n");

    // ─── Round-trip: Arrow -> compute -> Arrow ───
    println!("3. Round-trip: Arrow -> compute -> Arrow");

    let input = Float64Array::from(vec![2.0, 4.0, 6.0, 8.0]);
    println!("   Input:  {:?}", input.values());

    // Borrow as ndarray view, compute, transfer back
    let view = input.as_ndarray().unwrap();
    let doubled = &view * 2.0; // This is the only allocation (the computation result)
    let output: PrimitiveArray<Float64Type> = doubled.into_arrow().unwrap();
    println!("   Output: {:?}", output.values());
    println!("   (Only the computation allocated — the bridge was free)");
}
