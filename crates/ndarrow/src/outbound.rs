//! ndarray to Arrow conversions (zero-copy ownership transfer).
//!
//! This module provides conversions from owned ndarray arrays to Arrow arrays.
//! Ownership of the underlying buffer is transferred — no allocation occurs when
//! the ndarray is in standard (C-contiguous) layout.
//!
//! When the layout is non-standard, a contiguous copy is made automatically before
//! transferring ownership.
//!
//! # Allocation contract
//!
//! - Standard layout: **zero allocation** (ownership transfer only).
//! - Non-standard layout: **one allocation** (`as_standard_layout().into_owned()`), then transfer.

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, PrimitiveArray};
use arrow_schema::Field;
use ndarray::{Array1, Array2};

use crate::{
    element::{IntoScalarBuffer, NdarrowElement},
    error::NdarrowError,
};

/// Zero-copy ownership transfer from an ndarray array to an Arrow array.
///
/// Implementations consume the ndarray and produce an Arrow array. When the ndarray
/// is in standard (C-contiguous) layout, the underlying `Vec<T>` is moved directly
/// into Arrow's buffer — no data is touched. When the layout is non-standard, a
/// contiguous copy is made first.
///
/// # Allocation contract
///
/// - Standard layout: **zero allocation**.
/// - Non-standard layout: **one allocation** (layout normalization), then zero-copy transfer.
pub trait IntoArrow {
    /// The Arrow array type produced by this conversion.
    type ArrowArray;

    /// Transfer ownership of this ndarray into an Arrow array.
    ///
    /// Returns `Ok` on success. The conversion is infallible for standard-layout
    /// arrays; non-standard layout triggers an automatic copy.
    ///
    /// # Errors
    ///
    /// Returns an [`NdarrowError`] if Arrow-compatible shape metadata cannot be
    /// represented for the target array type.
    fn into_arrow(self) -> Result<Self::ArrowArray, NdarrowError>;
}

// ─── Array1<T> -> PrimitiveArray<T::ArrowType> ───

/// Transfers ownership of an `Array1<T>` into a `PrimitiveArray`.
///
/// # Zero-copy path
///
/// When the array is contiguous with offset 0 and no excess capacity,
/// `into_raw_vec()` yields the exact backing `Vec<T>`, which is moved into
/// `ScalarBuffer::from(vec)` — no bytes are copied.
///
/// # Allocation cases
///
/// - **Non-standard layout**: copies into a C-contiguous buffer first.
/// - **Non-zero offset** or **excess capacity**: copies the relevant slice.
impl<T> IntoArrow for Array1<T>
where
    T: NdarrowElement + IntoScalarBuffer,
{
    type ArrowArray = PrimitiveArray<T::ArrowType>;

    fn into_arrow(self) -> Result<Self::ArrowArray, NdarrowError> {
        let len = self.len();

        // Normalize to standard layout if necessary.
        let standard =
            if self.is_standard_layout() { self } else { self.as_standard_layout().into_owned() };

        let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
        let off = offset.unwrap_or(0);

        let vec = if off == 0 {
            // Zero-copy path: truncate excess capacity (no data copy), then transfer.
            raw_vec.truncate(len);
            raw_vec
        } else {
            // Rare: sliced array with non-zero offset. Copy the relevant range.
            raw_vec[off..off + len].to_vec()
        };

        let buffer = T::into_scalar_buffer(vec);
        Ok(PrimitiveArray::new(buffer, None))
    }
}

// ─── Array2<T> -> FixedSizeListArray ───

/// Transfers ownership of an `Array2<T>` into a `FixedSizeListArray`.
///
/// The 2D array with shape (M, N) becomes a `FixedSizeList<T>(N)` with M rows.
/// Each row in the ndarray becomes one fixed-size list element.
///
/// # Zero-copy path
///
/// When the array is in standard (C-contiguous / row-major) layout, the flat
/// backing buffer is already in the correct order for Arrow's `FixedSizeList`:
/// `[row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]`. The `Vec<T>` is
/// moved directly into `ScalarBuffer::from(vec)`.
///
/// # Allocation cases
///
/// - **Non-standard layout** (e.g., Fortran-contiguous): copies into C-contiguous first.
/// - **Non-zero offset** or **excess capacity**: copies the relevant slice.
impl<T> IntoArrow for Array2<T>
where
    T: NdarrowElement + IntoScalarBuffer,
{
    type ArrowArray = FixedSizeListArray;

    fn into_arrow(self) -> Result<Self::ArrowArray, NdarrowError> {
        let (m, n) = self.dim();
        let total = m * n;

        // Normalize to standard (C-contiguous / row-major) layout.
        let standard =
            if self.is_standard_layout() { self } else { self.as_standard_layout().into_owned() };

        let (mut raw_vec, offset) = standard.into_raw_vec_and_offset();
        let off = offset.unwrap_or(0);

        let vec = if off == 0 {
            raw_vec.truncate(total);
            raw_vec
        } else {
            raw_vec[off..off + total].to_vec()
        };

        let buffer = T::into_scalar_buffer(vec);
        let values_array = PrimitiveArray::<T::ArrowType>::new(buffer, None);

        let field = Arc::new(Field::new("item", T::data_type(), false));
        let value_length = i32::try_from(n).map_err(|_| NdarrowError::ShapeMismatch {
            message: format!("Array2 column count {n} exceeds Arrow i32 value_length limits"),
        })?;

        let fsl = FixedSizeListArray::new(field, value_length, Arc::new(values_array), None);

        Ok(fsl)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use arrow_array::{
        Array,
        types::{Float32Type, Float64Type},
    };
    use ndarray::{Array1, Array2, array};

    use super::*;

    // ─── Array1 -> PrimitiveArray tests ───

    #[test]
    fn array1_f64_into_arrow() {
        let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let prim = arr.into_arrow().unwrap();
        assert_eq!(prim.len(), 4);
        assert_abs_diff_eq!(prim.value(0), 1.0);
        assert_abs_diff_eq!(prim.value(3), 4.0);
    }

    #[test]
    fn array1_f32_into_arrow() {
        let arr = Array1::from_vec(vec![10.0_f32, 20.0, 30.0]);
        let prim = arr.into_arrow().unwrap();
        assert_eq!(prim.len(), 3);
        assert_abs_diff_eq!(prim.value(0), 10.0_f32);
        assert_abs_diff_eq!(prim.value(2), 30.0_f32);
    }

    #[test]
    fn array1_empty() {
        let arr = Array1::<f64>::from_vec(vec![]);
        let prim = arr.into_arrow().unwrap();
        assert_eq!(prim.len(), 0);
    }

    #[test]
    fn array1_single_element() {
        let arr = Array1::from_vec(vec![42.0_f64]);
        let prim = arr.into_arrow().unwrap();
        assert_eq!(prim.len(), 1);
        assert_abs_diff_eq!(prim.value(0), 42.0);
    }

    #[test]
    fn array1_no_nulls() {
        let arr = Array1::from_vec(vec![1.0_f64, 2.0]);
        let prim = arr.into_arrow().unwrap();
        assert_eq!(prim.null_count(), 0);
    }

    #[test]
    fn array1_roundtrip_f64() {
        use crate::inbound::AsNdarray;

        let original = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let arr = Array1::from_vec(original.clone());
        let arrow = arr.into_arrow().unwrap();
        let view = arrow.as_ndarray().unwrap();
        let observed = view.as_slice().expect("contiguous view expected");
        assert_eq!(observed.len(), original.len());
        for (actual, expected) in observed.iter().zip(original.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn array1_roundtrip_f32() {
        use crate::inbound::AsNdarray;

        let original = vec![1.0_f32, 2.0, 3.0];
        let arr = Array1::from_vec(original.clone());
        let arrow: PrimitiveArray<Float32Type> = arr.into_arrow().unwrap();
        let view = arrow.as_ndarray().unwrap();
        let observed = view.as_slice().expect("contiguous view expected");
        assert_eq!(observed.len(), original.len());
        for (actual, expected) in observed.iter().zip(original.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn array1_non_standard_layout() {
        // Create a non-contiguous array by slicing with stride
        let arr = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let sliced = arr.slice(ndarray::s![..;2]).to_owned(); // [1.0, 3.0, 5.0]
        // This should still work (auto-copies to standard layout)
        let prim = sliced.into_arrow().unwrap();
        assert_eq!(prim.len(), 3);
        assert_abs_diff_eq!(prim.value(0), 1.0);
        assert_abs_diff_eq!(prim.value(1), 3.0);
        assert_abs_diff_eq!(prim.value(2), 5.0);
    }

    // ─── Array2 -> FixedSizeListArray tests ───

    #[test]
    fn array2_f64_into_arrow() {
        let arr = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 3);
        assert_eq!(fsl.value_length(), 2);
    }

    #[test]
    fn array2_f32_into_arrow() {
        let arr = array![[10.0_f32, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_length(), 3);
    }

    #[test]
    fn array2_empty_rows() {
        let arr = Array2::<f64>::from_shape_vec((0, 3), vec![]).unwrap();
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 0);
        assert_eq!(fsl.value_length(), 3);
    }

    #[test]
    fn array2_single_row() {
        let arr = array![[10.0_f64, 20.0, 30.0]];
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 1);
        assert_eq!(fsl.value_length(), 3);
    }

    #[test]
    fn array2_single_column() {
        let arr = array![[1.0_f64], [2.0], [3.0]];
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 3);
        assert_eq!(fsl.value_length(), 1);
    }

    #[test]
    fn array2_no_nulls() {
        let arr = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.null_count(), 0);
    }

    #[test]
    fn array2_values_correct() {
        // Verify the flat buffer order matches row-major layout.
        let arr = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fsl = arr.into_arrow().unwrap();

        let inner = fsl.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
        let values: Vec<f64> = inner.values().iter().copied().collect();
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn array2_roundtrip_f64() {
        use arrow_array::types::Float64Type;

        use crate::inbound::fixed_size_list_as_array2;

        let original = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let expected = original.clone();

        let fsl = original.into_arrow().unwrap();
        let view = fixed_size_list_as_array2::<Float64Type>(&fsl).unwrap();

        assert_eq!(view.dim(), expected.dim());
        for (actual, expected) in view.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn array2_roundtrip_f32() {
        use arrow_array::types::Float32Type;

        use crate::inbound::fixed_size_list_as_array2;

        let original = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let expected = original.clone();

        let fsl = original.into_arrow().unwrap();
        let view = fixed_size_list_as_array2::<Float32Type>(&fsl).unwrap();

        assert_eq!(view.dim(), expected.dim());
        for (actual, expected) in view.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    #[test]
    fn array2_fortran_layout() {
        use ndarray::ShapeBuilder;

        // Create Fortran (column-major) layout — should auto-convert.
        let arr = Array2::from_shape_vec((2, 3).f(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).unwrap();
        assert!(!arr.is_standard_layout());

        let fsl = arr.into_arrow().unwrap();
        assert_eq!(fsl.len(), 2);
        assert_eq!(fsl.value_length(), 3);

        // Verify values are in row-major order after conversion.
        let inner = fsl.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
        let values: Vec<f64> = inner.values().iter().copied().collect();
        // Fortran layout stored: col-major [1, 3, 5, 2, 4, 6]
        // After standard layout conversion, row-major: [1, 2, 3, 4, 5, 6]
        // Row 0: [1, 2, 3], Row 1: [4, 5, 6]
        // Wait, let me think about this more carefully.
        // Original Fortran layout: shape (2, 3), data [1, 3, 5, 2, 4, 6]
        // In Fortran order (column-major):
        //   (0,0)=1, (1,0)=3, (0,1)=5, (1,1)=2, (0,2)=4, (1,2)=6
        // Wait, that doesn't seem right. Let me reconsider.
        //
        // Actually, `from_shape_vec((2,3).f(), [1,3,5,2,4,6])` with .f() flag means
        // Fortran-order (column-major) interpretation:
        //   Column 0: [1, 3], Column 1: [5, 2], Column 2: [4, 6]
        //   So logically: [[1, 5, 4], [3, 2, 6]]
        //
        // After as_standard_layout (row-major): [1, 5, 4, 3, 2, 6]
        let expected = [1.0, 5.0, 4.0, 3.0, 2.0, 6.0];
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected);
        }
    }

    // ─── Zero-copy verification ───

    #[test]
    fn array1_zero_copy_transfer() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let data_ptr = data.as_ptr();
        let arr = Array1::from_vec(data);

        let prim = arr.into_arrow().unwrap();
        let arrow_ptr = prim.values().as_ref().as_ptr();

        // The Arrow buffer should point to the same memory as the original Vec.
        assert_eq!(data_ptr, arrow_ptr, "ownership transfer must preserve the buffer pointer");
    }

    #[test]
    fn array2_zero_copy_transfer() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_ptr = data.as_ptr();
        let arr = Array2::from_shape_vec((2, 3), data).unwrap();

        let fsl = arr.into_arrow().unwrap();
        let inner = fsl.values().as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
        let arrow_ptr = inner.values().as_ref().as_ptr();

        // The Arrow buffer should point to the same memory as the original Vec.
        assert_eq!(data_ptr, arrow_ptr, "ownership transfer must preserve the buffer pointer");
    }
}
