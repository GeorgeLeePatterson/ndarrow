//! Error types for the Narrow library.

use thiserror::Error;

/// Result type alias for Narrow operations.
pub type Result<T> = std::result::Result<T, NarrowError>;

/// Errors that can occur in Narrow operations.
#[derive(Error, Debug)]
pub enum NarrowError {
    /// Dimension mismatch between vectors or operations.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Length mismatch between arrays in an operation.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Invalid array type for the operation.
    #[error("invalid array type: expected {expected}, got {actual}")]
    InvalidArrayType { expected: String, actual: String },

    /// Invalid vector dimension (e.g., zero or negative).
    #[error("invalid dimension: {0}")]
    InvalidDimension(usize),

    /// Operation on empty array where data is required.
    #[error("empty array: operation requires at least one element")]
    EmptyArray,

    /// Null values are not supported for this operation.
    #[error("null values not supported in operation: {0}")]
    NullNotSupported(String),

    /// Arrow error wrapper.
    #[error("arrow error: {0}")]
    ArrowError(#[from] arrow::error::ArrowError),

    /// General computation error.
    #[error("computation error: {0}")]
    ComputationError(String),
}
