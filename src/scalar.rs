//! Scalar types for vector elements.
//!
//! This module defines traits and types for scalar values that can be used as vector elements.
//! The primary constraint is that scalars must be floating-point types suitable for linear
//! algebra operations.

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Float32Type, Float64Type};
use std::fmt::Debug;

/// Trait for scalar types that can be used in vector operations.
///
/// This trait is sealed and can only be implemented for supported Arrow primitive types
/// (Float32, Float64, and potentially Float16 in the future).
pub trait VectorScalar:
    private::Sealed + ArrowPrimitiveType + Debug + Send + Sync + 'static
{
    /// The native Rust type for this scalar.
    type Native: num_traits::Float + Debug + Send + Sync + 'static;

    /// Convert from Arrow array to primitive array.
    fn as_primitive_array(array: &dyn Array) -> Option<&PrimitiveArray<Self>>;

    /// Check if this type matches the given Arrow array.
    fn matches_array(array: &dyn Array) -> bool;
}

impl VectorScalar for Float32Type {
    type Native = f32;

    fn as_primitive_array(array: &dyn Array) -> Option<&PrimitiveArray<Self>> {
        array.as_any().downcast_ref::<PrimitiveArray<Float32Type>>()
    }

    fn matches_array(array: &dyn Array) -> bool {
        matches!(array.data_type(), arrow::datatypes::DataType::Float32)
    }
}

impl VectorScalar for Float64Type {
    type Native = f64;

    fn as_primitive_array(array: &dyn Array) -> Option<&PrimitiveArray<Self>> {
        array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>()
    }

    fn matches_array(array: &dyn Array) -> bool {
        matches!(array.data_type(), arrow::datatypes::DataType::Float64)
    }
}

// Sealed trait pattern to prevent external implementations
mod private {
    use arrow::datatypes::{Float32Type, Float64Type};

    pub trait Sealed {}
    impl Sealed for Float32Type {}
    impl Sealed for Float64Type {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;

    #[test]
    fn test_scalar_type_checking() {
        let array = Float32Array::from(vec![1.0, 2.0, 3.0]);
        assert!(Float32Type::matches_array(&array));
        assert!(!Float64Type::matches_array(&array));
    }
}
