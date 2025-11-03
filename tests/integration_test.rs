//! Integration tests for the narrow library.
//!
//! These tests demonstrate end-to-end usage of the library, including:
//! - Creating vector arrays from various sources
//! - Zero-copy conversions to ndarray and nalgebra
//! - Vector arithmetic operations
//! - Similarity computations
//! - Round-trip conversions

use approx::assert_relative_eq;
use narrow::{DenseVectorArray, DenseVectorArrayF32, Result};

#[test]
fn test_create_and_query() -> Result<()> {
    // Create a vector database of embeddings
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],     // Pure X
        vec![0.0, 1.0, 0.0],     // Pure Y
        vec![0.0, 0.0, 1.0],     // Pure Z
        vec![0.707, 0.707, 0.0], // X+Y diagonal
    ];

    let vectors = DenseVectorArrayF32::from_vecs(&embeddings, 3)?;

    // Query with a vector close to X axis
    let query = vec![0.9, 0.1, 0.0];
    let similarities = vectors.cosine_similarity(&query)?;

    // First vector (pure X) should be most similar
    assert!(similarities[0] > similarities[1]);
    assert!(similarities[0] > similarities[2]);
    assert!(similarities[0] > similarities[3]);

    Ok(())
}

#[test]
fn test_vector_arithmetic() -> Result<()> {
    let a = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0], vec![3.0, 4.0]], 2)?;

    let b = DenseVectorArrayF32::from_vecs(&[vec![10.0, 20.0], vec![30.0, 40.0]], 2)?;

    // Test addition
    let sum = a.add(&b)?;
    assert_eq!(sum.get(0).unwrap(), vec![11.0, 22.0]);
    assert_eq!(sum.get(1).unwrap(), vec![33.0, 44.0]);

    // Test subtraction
    let diff = b.subtract(&a)?;
    assert_eq!(diff.get(0).unwrap(), vec![9.0, 18.0]);

    // Test scalar multiplication
    let scaled = a.scalar_multiply(2.0)?;
    assert_eq!(scaled.get(0).unwrap(), vec![2.0, 4.0]);

    // Test element-wise multiplication
    let product = a.multiply(&b)?;
    assert_eq!(product.get(0).unwrap(), vec![10.0, 40.0]);

    Ok(())
}

#[test]
fn test_similarity_metrics() -> Result<()> {
    let vectors =
        DenseVectorArrayF32::from_vecs(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]], 2)?;

    let query = vec![1.0, 0.0];

    // Cosine similarity
    let cosine = vectors.cosine_similarity(&query)?;
    assert_relative_eq!(cosine[0], 1.0, epsilon = 1e-6); // Identical
    assert_relative_eq!(cosine[1], 0.0, epsilon = 1e-6); // Orthogonal
    assert_relative_eq!(cosine[2], 0.707106781, epsilon = 1e-6); // 45 degrees

    // Dot product
    let dots = vectors.dot_product(&query)?;
    assert_relative_eq!(dots[0], 1.0);
    assert_relative_eq!(dots[1], 0.0);
    assert_relative_eq!(dots[2], 1.0);

    // Euclidean distance
    let euclidean = vectors.euclidean_distance(&query)?;
    assert_relative_eq!(euclidean[0], 0.0); // Identical
    assert_relative_eq!(euclidean[1], 1.41421356, epsilon = 1e-6); // sqrt(2)

    // Manhattan distance
    let manhattan = vectors.manhattan_distance(&query)?;
    assert_relative_eq!(manhattan[0], 0.0);
    assert_relative_eq!(manhattan[1], 2.0);
    assert_relative_eq!(manhattan[2], 1.0);

    Ok(())
}

#[test]
fn test_ndarray_integration() -> Result<()> {
    use ndarray::array;

    // Create from ndarray
    let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];

    let vectors = DenseVectorArrayF32::from_ndarray(&matrix.view())?;
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors.dimension(), 3);

    // Convert back to ndarray (zero-copy)
    let view = vectors.as_ndarray()?;
    assert_eq!(view.shape(), &[2, 3]);
    assert_relative_eq!(view[[0, 0]], 1.0);
    assert_relative_eq!(view[[1, 2]], 6.0);

    // Get single vector view
    let vec_view = vectors.ndarray_view(0)?;
    assert_eq!(vec_view.len(), 3);
    assert_relative_eq!(vec_view[0], 1.0);

    // Perform ndarray operations
    let sum = view.sum();
    assert_relative_eq!(sum, 21.0);

    Ok(())
}

#[test]
fn test_nalgebra_integration() -> Result<()> {
    use nalgebra::{DMatrix, DVector};

    // Create from nalgebra matrix
    let matrix = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let vectors = DenseVectorArrayF32::from_nalgebra_matrix(&matrix)?;
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors.dimension(), 3);

    // Convert back to nalgebra
    let na_matrix = vectors.to_nalgebra_matrix()?;
    assert_eq!(na_matrix.nrows(), 2);
    assert_eq!(na_matrix.ncols(), 3);
    assert_relative_eq!(na_matrix[(0, 0)], 1.0);

    // Get single vector
    let vec = vectors.to_nalgebra_vector(0)?;
    assert_eq!(vec.len(), 3);
    assert_relative_eq!(vec[0], 1.0);

    // Get view
    let view = vectors.nalgebra_view(1)?;
    assert_relative_eq!(view[0], 4.0);

    // Create from single vector
    let single_vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let single_array = DenseVectorArrayF32::from_nalgebra_vector(&single_vec)?;
    assert_eq!(single_array.len(), 1);
    assert_eq!(single_array.dimension(), 3);

    Ok(())
}

#[test]
fn test_vector_search_workflow() -> Result<()> {
    // Simulate a typical vector search workflow

    // 1. Create a database of embeddings (e.g., from a model)
    let database = vec![
        vec![0.1, 0.9, 0.5], // Document 1
        vec![0.8, 0.2, 0.3], // Document 2
        vec![0.3, 0.7, 0.6], // Document 3
        vec![0.9, 0.1, 0.2], // Document 4
        vec![0.2, 0.8, 0.4], // Document 5
    ];

    let vectors = DenseVectorArrayF32::from_vecs(&database, 3)?;

    // 2. Query vector (e.g., from user query)
    let query = vec![0.15, 0.85, 0.45];

    // 3. Compute similarities
    let similarities = vectors.cosine_similarity(&query)?;

    // 4. Find top-k results
    let mut indexed_similarities: Vec<(usize, f32)> = similarities
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();

    indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 5. Top result should be document 5 (index 4) or document 1 (index 0)
    // since they're closest to the query vector
    let top_idx = indexed_similarities[0].0;
    assert!(top_idx == 0 || top_idx == 4);

    // Verify similarity is high
    assert!(indexed_similarities[0].1 > 0.95);

    Ok(())
}

#[test]
fn test_batch_operations() -> Result<()> {
    // Simulate batch processing of vectors

    let vectors =
        DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]], 2)?;

    // Normalize all vectors (using scalar multiplication and norms)
    let norms = vectors.euclidean_distance(&vec![0.0, 0.0])?;

    // Create normalized versions
    let mut normalized_vecs = Vec::new();
    for i in 0..vectors.len() {
        let vec = vectors.get(i).unwrap();
        let norm = norms[i];
        let normalized: Vec<f32> = vec.iter().map(|&v| v / norm).collect();
        normalized_vecs.push(normalized);
    }

    let normalized = DenseVectorArrayF32::from_vecs(&normalized_vecs, 2)?;

    // All normalized vectors should have norm ~1.0
    let normalized_norms = normalized.euclidean_distance(&vec![0.0, 0.0])?;
    for norm in normalized_norms {
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    Ok(())
}

#[test]
fn test_type_safety() -> Result<()> {
    use narrow::DenseVectorArrayF64;

    // Float32 arrays
    let f32_array = DenseVectorArrayF32::from_vecs(&[vec![1.0, 2.0]], 2)?;

    // Float64 arrays
    let f64_array = DenseVectorArrayF64::from_vecs(&[vec![1.0, 2.0]], 2)?;

    // Both work with their respective types
    assert_eq!(f32_array.dimension(), 2);
    assert_eq!(f64_array.dimension(), 2);

    // Type system prevents mixing Float32 and Float64 operations at compile time
    // This would fail to compile:
    // let sum = f32_array.add(&f64_array)?;  // Compile error!

    Ok(())
}
