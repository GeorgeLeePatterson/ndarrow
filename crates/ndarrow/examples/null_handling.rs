//! Null handling with the three-tier API.
//!
//! Demonstrates the three approaches to null handling:
//! 1. Validated: returns `Err` if nulls present (safest)
//! 2. Unchecked: caller guarantees no nulls (fastest)
//! 3. Masked: returns view + validity bitmap (most flexible)
//!
//! Run with: `cargo run -p ndarrow --example null_handling`

use arrow_array::Float64Array;
use ndarrow::AsNdarray;

fn main() {
    println!("=== ndarrow: Null Handling Example ===\n");

    // ─── Array with no nulls ───
    let clean = Float64Array::from(vec![1.0, 2.0, 3.0]);

    // ─── Array with nulls ───
    let nullable = Float64Array::from(vec![Some(10.0), None, Some(30.0), None, Some(50.0)]);

    // ─── Tier 1: Validated ───
    println!("1. Validated (as_ndarray) — returns Result");

    match clean.as_ndarray() {
        Ok(view) => println!("   Clean array: {view} (OK)"),
        Err(e) => println!("   Clean array: Error: {e}"),
    }

    match nullable.as_ndarray() {
        Ok(view) => println!("   Nullable array: {view} (OK)"),
        Err(e) => println!("   Nullable array: Error: {e}"),
    }
    println!();

    // ─── Tier 2: Unchecked ───
    println!("2. Unchecked (as_ndarray_unchecked) — caller guarantees no nulls");

    let view = unsafe { clean.as_ndarray_unchecked() };
    println!("   Clean array (unchecked): {view}");
    println!("   (Zero cost — no null check performed)");
    // NOTE: calling this on `nullable` would be UB. Don't do it.
    println!();

    // ─── Tier 3: Masked ───
    println!("3. Masked (as_ndarray_masked) — returns view + validity bitmap");

    let (view, mask) = nullable.as_ndarray_masked();
    println!("   View (all positions): {view}");

    if let Some(nulls) = mask {
        print!("   Validity bitmap:      [");
        for i in 0..view.len() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", if nulls.is_valid(i) { "valid" } else { "NULL" });
        }
        println!("]");

        // Use the mask to compute with only valid values.
        let valid_sum: f64 =
            view.iter().enumerate().filter(|(i, _)| nulls.is_valid(*i)).map(|(_, v)| v).sum();
        let valid_count = nulls.len() - nulls.null_count();
        let valid_count_u32 =
            u32::try_from(valid_count).expect("valid_count must fit in u32 for mean calculation");
        let mean = valid_sum / f64::from(valid_count_u32);

        println!("   Sum of valid values: {valid_sum}");
        println!("   Mean of valid values: {mean}");
    } else {
        println!("   No nulls — mask is None");
    }
}
