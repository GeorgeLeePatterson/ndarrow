//! Narrow is a Rust library for running linear algebra operations with Arrow interop.
//!
//! This library provides a semantic type system for linear algebra operations on Apache Arrow
//! arrays, with zero-copy conversions to ndarray and nalgebra. It's designed to bridge the gap
//! between Arrow's columnar data model and linear algebra operations, particularly for use cases
//! like vector similarity search in DataFusion.
//!
//! # Type System
//!
//! Narrow provides wrapper types around Arrow arrays that encode vector semantics:
//!
//! - **DenseVectorArray**: Dense vectors stored as `FixedSizeListArray<T>` where T is Float32/Float64
//! - **SparseVectorArray** (future): Sparse vectors with efficient storage
//!
//! # Type Compatibility
//!
//! Operations between different vector types follow strict compatibility rules:
//!
//! - Dense vectors can be added/multiplied with other dense vectors of the same dimension
//! - Scalar multiplication is always valid
//! - Similarity operations require matching dimensions
//!
//! # Example
//!
//! ```rust,ignore
//! use narrow::DenseVectorArray;
//! use arrow::array::Float32Array;
//!
//! // Create a dense vector array from Arrow data
//! let vectors = DenseVectorArray::from_arrow(fixed_size_list)?;
//!
//! // Zero-copy conversion to ndarray for batch operations
//! let matrix = vectors.as_ndarray()?;
//!
//! // Vector arithmetic
//! let sum = vectors.add(&other_vectors)?;
//!
//! // Similarity operations
//! let similarities = vectors.cosine_similarity(&query_vector)?;
//! ```

pub mod conversions;
pub mod dense;
pub mod error;
pub mod ops;
pub mod scalar;

pub use dense::DenseVectorArray;
pub use error::{NarrowError, Result};
pub use scalar::VectorScalar;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Basic smoke test
        assert!(true);
    }
}
