//! Element type bridge between Arrow primitive types and ndarray element types.
//!
//! The [`NdarrowElement`] trait connects Arrow's type system with ndarray's element
//! requirements. Any type implementing this trait can be used in both Arrow arrays
//! and ndarray arrays, enabling zero-copy conversions between the two.

use arrow_array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;

/// Bridges Arrow's [`ArrowPrimitiveType`] with ndarray's element requirements.
///
/// A type implementing `NdarrowElement` can appear in both Arrow `PrimitiveArray<T>`
/// and ndarray `Array<T, D>`, enabling zero-copy conversions between the two.
///
/// # Implementations
///
/// Concrete implementations exist for:
/// - `f32` (via `Float32Type`)
/// - `f64` (via `Float64Type`)
///
/// Additional types can be added by implementing this trait.
///
/// # Does not allocate
///
/// This trait defines a compile-time mapping. It introduces no runtime cost.
pub trait NdarrowElement:
    Copy + 'static + std::fmt::Debug + num_traits::Zero + arrow_buffer::ArrowNativeType
{
    /// The corresponding Arrow primitive type.
    type ArrowType: ArrowPrimitiveType<Native = Self>;

    /// The Arrow [`DataType`] for this element.
    #[must_use]
    fn data_type() -> DataType {
        Self::ArrowType::DATA_TYPE
    }
}

impl NdarrowElement for f32 {
    type ArrowType = Float32Type;
}

impl NdarrowElement for f64 {
    type ArrowType = Float64Type;
}

/// Extension trait for converting a `Vec<T>` into an Arrow `ScalarBuffer<T>`.
///
/// This enables zero-copy ownership transfer from ndarray's internal `Vec<T>`
/// to Arrow's buffer representation.
///
/// # Does not allocate
///
/// `ScalarBuffer::from(vec)` takes ownership of the `Vec<T>` without copying.
pub(crate) trait IntoScalarBuffer: NdarrowElement {
    fn into_scalar_buffer(vec: Vec<Self>) -> ScalarBuffer<Self>;
}

impl IntoScalarBuffer for f32 {
    fn into_scalar_buffer(vec: Vec<Self>) -> ScalarBuffer<Self> {
        ScalarBuffer::from(vec)
    }
}

impl IntoScalarBuffer for f64 {
    fn into_scalar_buffer(vec: Vec<Self>) -> ScalarBuffer<Self> {
        ScalarBuffer::from(vec)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use arrow_schema::DataType;

    use super::*;

    #[test]
    fn f32_data_type() {
        assert_eq!(f32::data_type(), DataType::Float32);
    }

    #[test]
    fn f64_data_type() {
        assert_eq!(f64::data_type(), DataType::Float64);
    }

    #[test]
    fn f32_arrow_type_native() {
        // Verify the associated type mapping is correct at the type level.
        fn assert_native<T: NdarrowElement>()
        where
            <T::ArrowType as ArrowPrimitiveType>::Native: Into<f64>,
        {
        }
        assert_native::<f32>();
    }

    #[test]
    fn f64_arrow_type_native() {
        fn assert_native<T: NdarrowElement>()
        where
            <T::ArrowType as ArrowPrimitiveType>::Native: Into<f64>,
        {
        }
        assert_native::<f64>();
    }

    #[test]
    fn into_scalar_buffer_f32() {
        let vec = vec![1.0_f32, 2.0, 3.0];
        let buf = f32::into_scalar_buffer(vec);
        assert_eq!(buf.len(), 3);
        assert_abs_diff_eq!(buf[0], 1.0_f32);
        assert_abs_diff_eq!(buf[2], 3.0_f32);
    }

    #[test]
    fn into_scalar_buffer_f64() {
        let vec = vec![1.0_f64, 2.0, 3.0];
        let buf = f64::into_scalar_buffer(vec);
        assert_eq!(buf.len(), 3);
        assert_abs_diff_eq!(buf[0], 1.0_f64);
        assert_abs_diff_eq!(buf[2], 3.0_f64);
    }

    #[test]
    fn into_scalar_buffer_empty() {
        let vec: Vec<f64> = vec![];
        let buf = f64::into_scalar_buffer(vec);
        assert_eq!(buf.len(), 0);
    }
}
