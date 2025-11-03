//! Zero-copy conversions between Arrow arrays and linear algebra libraries.
//!
//! This module provides efficient conversions to and from:
//! - `ndarray`: For general numeric computing and SIMD operations
//! - `nalgebra`: For linear algebra operations

pub mod ndarray_conv;
pub mod nalgebra_conv;

pub use ndarray_conv::*;
pub use nalgebra_conv::*;
