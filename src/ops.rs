//! Vector operations including arithmetic and similarity metrics.
//!
//! This module provides implementations of common vector operations:
//! - Arithmetic: addition, subtraction, scalar multiplication
//! - Similarity: cosine similarity, dot product, Euclidean distance

pub mod arithmetic;
pub mod similarity;

pub use arithmetic::*;
pub use similarity::*;
