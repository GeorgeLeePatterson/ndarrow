# 🧮 Narrow

**Linear algebra for Apache Arrow data**

Narrow is a Rust library that bridges Apache Arrow's columnar data format with linear algebra operations. It provides a semantic type system for vector operations, zero-copy conversions to ndarray and nalgebra, and high-performance vector similarity computations optimized for use cases like vector search in DataFusion.

## Features

- **Type-safe vector arrays**: Wrapper types around Arrow's `FixedSizeListArray` with compile-time type safety and runtime dimension validation
- **Zero-copy conversions**: Efficient conversions to ndarray and nalgebra without data copying
- **Vector arithmetic**: Element-wise addition, subtraction, scalar multiplication, and Hadamard products
- **Similarity metrics**: Cosine similarity, dot product, Euclidean distance, and Manhattan distance
- **Semantic type system**: Clear rules about operation compatibility between vector types
- **DataFusion integration**: (Planned) UDFs for vector operations in SQL queries

## Motivation

Apache Arrow has tensor types in its schema, but these are designed for the IPC/serialization layer (e.g., ML model weights), not for columnar storage or compute operations. For vector similarity search and linear algebra on tabular data, the standard approach is to use `FixedSizeList<Float32>` or `FixedSizeList<Float64>`.

Narrow provides a clean abstraction over this pattern, enabling:
- Natural integration with DataFusion queries
- Compatibility with Arrow's compute kernels
- Proper columnar representation
- Bridge to linear algebra libraries (ndarray, nalgebra)

## Example Usage

```rust
use narrow::{DenseVectorArrayF32, Result};

fn main() -> Result<()> {
    // Create a vector array from raw data
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],  // Document 1 embedding
        vec![0.0, 1.0, 0.0],  // Document 2 embedding
        vec![0.707, 0.707, 0.0],  // Document 3 embedding
    ];

    let vectors = DenseVectorArrayF32::from_vecs(&embeddings, 3)?;

    // Perform similarity search
    let query = vec![0.9, 0.1, 0.0];
    let similarities = vectors.cosine_similarity(&query)?;

    // Find most similar vector
    let (best_idx, best_score) = similarities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Most similar document: {} (score: {:.3})", best_idx, best_score);

    // Vector arithmetic
    let other = DenseVectorArrayF32::from_vecs(&[
        vec![1.0, 1.0, 1.0],
    ], 3)?;

    let sum = vectors.add(&other)?;
    let scaled = vectors.scalar_multiply(2.0)?;

    // Zero-copy conversion to ndarray
    let matrix = vectors.as_ndarray()?;
    println!("Matrix shape: {:?}", matrix.shape());

    // Convert to nalgebra for linear algebra
    let na_matrix = vectors.to_nalgebra_matrix()?;

    Ok(())
}
```

## Type System

Narrow provides a semantic type system for vector operations:

### Dense Vectors

- **`DenseVectorArray<T>`**: Dense vectors backed by `FixedSizeListArray<T>`
- Type parameter `T` is `Float32Type`, `Float64Type`, or (future) `Float16Type`
- Type aliases: `DenseVectorArrayF32`, `DenseVectorArrayF64`

### Type Compatibility Rules

Operations between vectors must satisfy:

1. **Same dimension**: All arithmetic and similarity operations require matching dimensions
2. **Same length**: Binary operations require arrays with the same number of vectors
3. **Same scalar type**: Cannot mix Float32 and Float64 operations (enforced at compile time)

### Future Types

- **`SparseVectorArray`**: Efficient storage for sparse vectors
- **`QuantizedVectorArray`**: Quantized vectors for memory-efficient search

## Operations

### Arithmetic Operations

- `add(&self, other: &Self) -> Result<Self>`: Element-wise addition
- `subtract(&self, other: &Self) -> Result<Self>`: Element-wise subtraction
- `multiply(&self, other: &Self) -> Result<Self>`: Element-wise (Hadamard) product
- `scalar_multiply(&self, scalar: T) -> Result<Self>`: Scalar multiplication

### Similarity Metrics

- `cosine_similarity(&self, query: &[T]) -> Result<Vec<T>>`: Cosine similarity
- `dot_product(&self, query: &[T]) -> Result<Vec<T>>`: Dot product (inner product)
- `euclidean_distance(&self, query: &[T]) -> Result<Vec<T>>`: L2 distance
- `manhattan_distance(&self, query: &[T]) -> Result<Vec<T>>`: L1 distance

### Conversions

**ndarray**:
- `as_ndarray(&self) -> Result<ArrayView2<T>>`: Zero-copy 2D view
- `ndarray_view(&self, index: usize) -> Result<ArrayView1<T>>`: Zero-copy 1D view
- `from_ndarray(array: &ArrayView2<T>) -> Result<Self>`: Create from ndarray

**nalgebra**:
- `to_nalgebra_matrix(&self) -> Result<DMatrix<T>>`: Convert to DMatrix
- `to_nalgebra_vector(&self, index: usize) -> Result<DVector<T>>`: Convert single vector
- `nalgebra_view(&self, index: usize) -> Result<DVectorView<T>>`: Zero-copy view
- `from_nalgebra_matrix(matrix: &DMatrix<T>) -> Result<Self>`: Create from DMatrix
- `from_nalgebra_vector(vector: &DVector<T>) -> Result<Self>`: Create from DVector

## Architecture

### Core Modules

- **`dense`**: Dense vector array implementation
- **`scalar`**: Scalar type traits for Float32/Float64/Float16
- **`ops`**: Vector operations
  - `ops::arithmetic`: Vector arithmetic operations
  - `ops::similarity`: Similarity and distance metrics
- **`conversions`**: Zero-copy conversions
  - `conversions::ndarray_conv`: ndarray integration
  - `conversions::nalgebra_conv`: nalgebra integration
- **`error`**: Error types and result aliases

### Design Principles

1. **Safety first**: All unsafe operations are encapsulated in safe APIs
2. **Zero-copy where possible**: Use views and slices to avoid data copying
3. **Type-driven semantics**: Encode vector semantics in the type system
4. **Arrow native**: Leverage Arrow's columnar strengths (SIMD, nulls, compute kernels)
5. **Composable**: Works seamlessly with DataFusion, arrow-rs, and linear algebra libraries

## Use Cases

### Vector Similarity Search

```rust
// Create database of embeddings
let db_vectors = DenseVectorArrayF32::from_vecs(&embeddings, dimension)?;

// Query
let query = get_query_embedding();
let scores = db_vectors.cosine_similarity(&query)?;

// Get top-k results
let top_k = get_top_k_indices(&scores, k);
```

### DataFusion Integration (Planned)

```sql
-- Register vector similarity UDF
SELECT
    id,
    cosine_similarity(embedding, $query_vector) AS score
FROM
    documents
WHERE
    score > 0.7
ORDER BY
    score DESC
LIMIT 10;
```

### Linear Algebra Pipelines

```rust
// Zero-copy to ndarray for batch operations
let matrix = vectors.as_ndarray()?;

// Perform computations
let normalized = normalize_rows(&matrix);

// Convert back to Arrow for storage/DataFusion
let result = DenseVectorArrayF32::from_ndarray(&normalized)?;
```

### Bridge to qdrant-DataFusion

```rust
// Convert qdrant vectors to narrow
let narrow_vectors = convert_qdrant_to_narrow(&qdrant_points)?;

// Perform operations in DataFusion
let df = ctx.read_table(narrow_vectors.to_arrow_table()?)?
    .filter(col("score").gt(lit(0.8)))?
    .limit(0, Some(100))?;
```

## Roadmap

### Phase 1: Core Foundation ✅
- [x] DenseVectorArray type
- [x] Vector arithmetic operations
- [x] Similarity metrics
- [x] ndarray conversions
- [x] nalgebra conversions
- [x] Comprehensive tests

### Phase 2: Performance (Next)
- [ ] SIMD-optimized kernels for similarity metrics
- [ ] Batch operations using BLAS/LAPACK
- [ ] Parallel computation support
- [ ] Benchmarking suite

### Phase 3: DataFusion Integration
- [ ] Scalar UDFs for similarity metrics
- [ ] Aggregate UDFs (e.g., vector centroid)
- [ ] Table-valued functions
- [ ] Query optimization hints

### Phase 4: Extended Types
- [ ] SparseVectorArray
- [ ] Float16 support
- [ ] Quantized vectors (int8, binary)
- [ ] Complex vectors

### Phase 5: Advanced Operations
- [ ] Matrix multiplication
- [ ] PCA / dimensionality reduction
- [ ] Vector quantization
- [ ] Approximate nearest neighbor structures

## Testing

```bash
cargo test
```

The library includes comprehensive unit tests in each module and integration tests demonstrating end-to-end workflows.

## Contributing

Contributions are welcome! Areas of particular interest:

- SIMD optimizations for similarity metrics
- Additional similarity/distance metrics
- DataFusion UDF implementations
- Sparse vector support
- Performance benchmarks

## License

Apache-2.0

## Acknowledgments

- **Apache Arrow**: For the excellent columnar data format
- **ndarray**: For efficient array operations
- **nalgebra**: For linear algebra primitives
- **DataFusion**: For query engine integration inspiration
