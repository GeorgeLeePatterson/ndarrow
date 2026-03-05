//! Error types for ndarrow conversions.

use arrow_schema::ArrowError;

/// Errors that can occur during Arrow <-> ndarray conversions.
#[derive(Debug)]
pub enum NdarrowError {
    /// Arrow array contains null values where none were expected.
    NullsPresent {
        /// Number of null values found.
        null_count: usize,
    },

    /// The Arrow array's data type does not match the expected type.
    TypeMismatch {
        /// Human-readable description of the mismatch.
        message: String,
    },

    /// The array shape is incompatible with the requested ndarray shape.
    ShapeMismatch {
        /// Human-readable description of the mismatch.
        message: String,
    },

    /// Extension type metadata is missing or invalid.
    InvalidMetadata {
        /// Human-readable description of the problem.
        message: String,
    },

    /// The inner array of a composite type (`FixedSizeList`, `Struct`) is not the expected type.
    InnerTypeMismatch {
        /// Human-readable description of the mismatch.
        message: String,
    },

    /// The ndarray is not in standard (C-contiguous) layout and cannot be transferred to Arrow
    /// without copying.
    NonStandardLayout,

    /// Sparse arrays have mismatched offsets between indices and values columns.
    SparseOffsetMismatch,

    /// An error propagated from the arrow crate.
    Arrow(ArrowError),

    /// An ndarray shape error.
    Shape(ndarray::ShapeError),
}

impl std::fmt::Display for NdarrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullsPresent { null_count } => {
                write!(f, "array contains {null_count} null value(s)")
            }
            Self::TypeMismatch { message } => write!(f, "type mismatch: {message}"),
            Self::ShapeMismatch { message } => write!(f, "shape mismatch: {message}"),
            Self::InvalidMetadata { message } => write!(f, "invalid metadata: {message}"),
            Self::InnerTypeMismatch { message } => write!(f, "inner type mismatch: {message}"),
            Self::NonStandardLayout => write!(
                f,
                "array is not in standard (C-contiguous) layout; \
                 cannot transfer to Arrow without copying"
            ),
            Self::SparseOffsetMismatch => {
                write!(f, "sparse indices and values columns have mismatched offsets")
            }
            Self::Arrow(e) => write!(f, "arrow error: {e}"),
            Self::Shape(e) => write!(f, "shape error: {e}"),
        }
    }
}

impl std::error::Error for NdarrowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arrow(e) => Some(e),
            Self::Shape(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ArrowError> for NdarrowError {
    fn from(err: ArrowError) -> Self {
        Self::Arrow(err)
    }
}

impl From<ndarray::ShapeError> for NdarrowError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Shape(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_nulls_present() {
        let err = NdarrowError::NullsPresent { null_count: 3 };
        assert_eq!(err.to_string(), "array contains 3 null value(s)");
    }

    #[test]
    fn display_type_mismatch() {
        let err = NdarrowError::TypeMismatch { message: "expected Float64, got Int32".into() };
        assert!(err.to_string().contains("expected Float64"));
    }

    #[test]
    fn display_shape_mismatch() {
        let err = NdarrowError::ShapeMismatch { message: "expected [3, 4], got [12]".into() };
        assert!(err.to_string().contains("expected [3, 4]"));
    }

    #[test]
    fn display_invalid_metadata() {
        let err = NdarrowError::InvalidMetadata { message: "missing shape key".into() };
        assert!(err.to_string().contains("missing shape key"));
    }

    #[test]
    fn display_inner_type_mismatch() {
        let err = NdarrowError::InnerTypeMismatch { message: "inner array is not Float64".into() };
        assert!(err.to_string().contains("inner array is not Float64"));
    }

    #[test]
    fn display_non_standard_layout() {
        let err = NdarrowError::NonStandardLayout;
        assert!(err.to_string().contains("C-contiguous"));
    }

    #[test]
    fn display_sparse_offset_mismatch() {
        let err = NdarrowError::SparseOffsetMismatch;
        assert!(err.to_string().contains("mismatched offsets"));
    }

    #[test]
    fn from_arrow_error() {
        let arrow_err = ArrowError::InvalidArgumentError("test".into());
        let err: NdarrowError = arrow_err.into();
        assert!(matches!(err, NdarrowError::Arrow(_)));
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn from_shape_error() {
        let shape_err =
            ndarray::ArrayView1::<f64>::from_shape(ndarray::Ix1(5), &[1.0, 2.0]).unwrap_err();
        let err: NdarrowError = shape_err.into();
        assert!(matches!(err, NdarrowError::Shape(_)));
    }

    #[test]
    fn error_source_arrow() {
        use std::error::Error;
        let err = NdarrowError::Arrow(ArrowError::InvalidArgumentError("src".into()));
        assert!(err.source().is_some());
    }

    #[test]
    fn error_source_none() {
        use std::error::Error;
        let err = NdarrowError::NonStandardLayout;
        assert!(err.source().is_none());
    }
}
