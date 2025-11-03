//! Similarity metrics for dense vector arrays.
//!
//! This module provides common similarity and distance metrics used in vector search,
//! including cosine similarity, dot product, and Euclidean distance.

use crate::dense::DenseVectorArray;
use crate::error::{NarrowError, Result};
use crate::scalar::VectorScalar;
use arrow::array::PrimitiveArray;
use num_traits::Float;

impl<T: VectorScalar> DenseVectorArray<T> {
    /// Compute dot products between vectors in this array and a query vector.
    ///
    /// Returns an array of dot products, one for each vector in this array.
    ///
    /// # Arguments
    ///
    /// - `query`: A slice representing the query vector
    ///
    /// # Errors
    ///
    /// Returns an error if the query vector dimension doesn't match this array's dimension.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 0.0],
    ///     vec![0.0, 1.0],
    /// ], 2)?;
    /// let query = vec![1.0, 1.0];
    /// let dots = vectors.dot_product(&query)?;
    /// // dots = [1.0, 1.0]
    /// ```
    pub fn dot_product(&self, query: &[T::Native]) -> Result<Vec<T::Native>> {
        if query.len() != self.dimension {
            return Err(NarrowError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let values = self.values();
        let mut results = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let dot: T::Native = (start..end)
                .zip(query.iter())
                .map(|(idx, &q)| values.value(idx) * q)
                .fold(T::Native::zero(), |acc, x| acc + x);

            results.push(dot);
        }

        Ok(results)
    }

    /// Compute cosine similarity between vectors in this array and a query vector.
    ///
    /// Cosine similarity is computed as: dot(a, b) / (norm(a) * norm(b))
    /// Values range from -1 (opposite) to 1 (identical direction).
    ///
    /// # Arguments
    ///
    /// - `query`: A slice representing the query vector
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query vector dimension doesn't match this array's dimension
    /// - Any vector or the query has zero norm (undefined similarity)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 0.0],
    ///     vec![1.0, 1.0],
    /// ], 2)?;
    /// let query = vec![1.0, 0.0];
    /// let similarities = vectors.cosine_similarity(&query)?;
    /// // similarities = [1.0, 0.707...]
    /// ```
    pub fn cosine_similarity(&self, query: &[T::Native]) -> Result<Vec<T::Native>> {
        if query.len() != self.dimension {
            return Err(NarrowError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // Compute query norm once
        let query_norm = query
            .iter()
            .map(|&x| x * x)
            .fold(T::Native::zero(), |acc, x| acc + x)
            .sqrt();

        if query_norm == T::Native::zero() {
            return Err(NarrowError::ComputationError(
                "query vector has zero norm".to_string(),
            ));
        }

        let values = self.values();
        let mut results = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let start = i * self.dimension;
            let end = start + self.dimension;

            // Compute dot product and norm simultaneously
            let (dot, norm_sq) = (start..end)
                .zip(query.iter())
                .map(|(idx, &q)| {
                    let v = values.value(idx);
                    (v * q, v * v)
                })
                .fold(
                    (T::Native::zero(), T::Native::zero()),
                    |(acc_dot, acc_norm), (d, n)| (acc_dot + d, acc_norm + n),
                );

            let norm = norm_sq.sqrt();

            if norm == T::Native::zero() {
                return Err(NarrowError::ComputationError(format!(
                    "vector at index {} has zero norm",
                    i
                )));
            }

            let similarity = dot / (norm * query_norm);
            results.push(similarity);
        }

        Ok(results)
    }

    /// Compute Euclidean (L2) distance between vectors in this array and a query vector.
    ///
    /// # Arguments
    ///
    /// - `query`: A slice representing the query vector
    ///
    /// # Errors
    ///
    /// Returns an error if the query vector dimension doesn't match this array's dimension.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let vectors = DenseVectorArrayF32::from_vecs(&[
    ///     vec![1.0, 0.0],
    ///     vec![0.0, 1.0],
    /// ], 2)?;
    /// let query = vec![0.0, 0.0];
    /// let distances = vectors.euclidean_distance(&query)?;
    /// // distances = [1.0, 1.0]
    /// ```
    pub fn euclidean_distance(&self, query: &[T::Native]) -> Result<Vec<T::Native>> {
        if query.len() != self.dimension {
            return Err(NarrowError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let values = self.values();
        let mut results = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let dist_sq: T::Native = (start..end)
                .zip(query.iter())
                .map(|(idx, &q)| {
                    let diff = values.value(idx) - q;
                    diff * diff
                })
                .fold(T::Native::zero(), |acc, x| acc + x);

            results.push(dist_sq.sqrt());
        }

        Ok(results)
    }

    /// Compute Manhattan (L1) distance between vectors in this array and a query vector.
    ///
    /// # Arguments
    ///
    /// - `query`: A slice representing the query vector
    ///
    /// # Errors
    ///
    /// Returns an error if the query vector dimension doesn't match this array's dimension.
    pub fn manhattan_distance(&self, query: &[T::Native]) -> Result<Vec<T::Native>> {
        if query.len() != self.dimension {
            return Err(NarrowError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let values = self.values();
        let mut results = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let dist: T::Native = (start..end)
                .zip(query.iter())
                .map(|(idx, &q)| (values.value(idx) - q).abs())
                .fold(T::Native::zero(), |acc, x| acc + x);

            results.push(dist);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::DenseVectorArrayF32;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product() {
        let vectors =
            DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], 3).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let dots = vectors.dot_product(&query).unwrap();

        assert_relative_eq!(dots[0], 1.0);
        assert_relative_eq!(dots[1], 4.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let vectors =
            DenseVectorArrayF32::from_vecs(&[vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0]], 2)
                .unwrap();

        let query = vec![1.0, 0.0];
        let similarities = vectors.cosine_similarity(&query).unwrap();

        assert_relative_eq!(similarities[0], 1.0);
        assert_relative_eq!(similarities[1], 0.707106781, epsilon = 1e-6);
        assert_relative_eq!(similarities[2], 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let vectors = DenseVectorArrayF32::from_vecs(&[vec![0.0, 0.0], vec![3.0, 4.0]], 2).unwrap();

        let query = vec![0.0, 0.0];
        let distances = vectors.euclidean_distance(&query).unwrap();

        assert_relative_eq!(distances[0], 0.0);
        assert_relative_eq!(distances[1], 5.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let vectors = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0], vec![4.0, 6.0]], 2).unwrap();

        let query = vec![0.0, 0.0];
        let distances = vectors.manhattan_distance(&query).unwrap();

        assert_relative_eq!(distances[0], 3.0);
        assert_relative_eq!(distances[1], 10.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let vectors = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2).unwrap();
        let query = vec![1.0, 2.0, 3.0];

        let result = vectors.dot_product(&query);
        assert!(matches!(result, Err(NarrowError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_zero_norm_error() {
        let vectors = DenseVectorArrayF32::from_vecs(&[vec![0.0, 0.0]], 2).unwrap();

        let query = vec![1.0, 0.0];
        let result = vectors.cosine_similarity(&query);
        assert!(matches!(result, Err(NarrowError::ComputationError(_))));
    }
}
