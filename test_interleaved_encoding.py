"""
Test script to verify the interleaved encoding maintains proper monotonicity.
"""
import numpy as np
from partition_optimization import FuzzyPartitionOptimizer

def test_interleaved_encoding():
    """Test that the interleaved encoding ensures monotonicity."""
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = np.random.randint(0, 3, 100)
    
    # Initialize optimizer
    optimizer = FuzzyPartitionOptimizer(
        optimization_method='separability',
        search_strategy='grid',
        verbose=False
    )
    
    # Test encoding from quantiles
    quantiles = [0, 20, 40, 60, 80, 100]
    X_feature = X[:, 0]
    
    # Encode
    encoded = optimizer._encode_partitions_from_quantiles(X_feature, quantiles)
    print(f"Encoded parameters ({len(encoded)} values):")
    print(encoded)
    
    # Decode
    partition_params = optimizer._decode_partitions(encoded, X_feature)
    print(f"\nDecoded trapezoids (3 x 4):")
    print(partition_params)
    
    # Verify monotonicity
    print("\n" + "="*80)
    print("MONOTONICITY VERIFICATION")
    print("="*80)
    
    low = partition_params[0]
    med = partition_params[1]
    high = partition_params[2]
    
    print(f"\nLow trapezoid:    [{low[0]:.3f}, {low[1]:.3f}, {low[2]:.3f}, {low[3]:.3f}]")
    print(f"Medium trapezoid: [{med[0]:.3f}, {med[1]:.3f}, {med[2]:.3f}, {med[3]:.3f}]")
    print(f"High trapezoid:   [{high[0]:.3f}, {high[1]:.3f}, {high[2]:.3f}, {high[3]:.3f}]")
    
    # Check ordering constraints
    print("\nOrdering checks:")
    checks = [
        ("Low[a] ≤ Low[b]", low[0] <= low[1]),
        ("Low[b] ≤ Low[c]", low[1] <= low[2]),
        ("Low[c] ≤ Medium[e]", low[2] <= med[0]),
        ("Medium[e] ≤ Low[d]", med[0] <= low[3]),
        ("Low[d] ≤ Medium[f]", low[3] <= med[1]),
        ("Medium[f] ≤ Medium[g]", med[1] <= med[2]),
        ("Medium[g] ≤ High[i]", med[2] <= high[0]),
        ("High[i] ≤ Medium[h]", high[0] <= med[3]),
        ("Medium[h] ≤ High[j]", med[3] <= high[1]),
        ("High[j] ≤ High[k]", high[1] <= high[2]),
        ("High[k] ≤ High[l]", high[2] <= high[3]),
    ]
    
    all_pass = True
    for constraint, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {constraint}")
        if not result:
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL MONOTONICITY CHECKS PASSED!")
        print("The interleaved encoding ensures proper ordering.")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("The encoding may not guarantee monotonicity correctly.")
    print("="*80)
    
    # Test with random positive increments
    print("\n\nTesting with random positive increments...")
    random_encoded = np.abs(np.random.randn(12))  # All positive
    random_encoded[0] = np.min(X_feature)  # Set starting point
    
    partition_params_random = optimizer._decode_partitions(random_encoded, X_feature)
    print(f"Random encoded: {random_encoded}")
    print(f"Decoded trapezoids:\n{partition_params_random}")
    
    # Flatten and check global monotonicity
    all_points = []
    for i, trap in enumerate(partition_params_random):
        for j, val in enumerate(trap):
            all_points.append((val, f"Trapezoid{i}[{j}]"))
    
    print("\nAll decoded points in order:")
    for val, name in all_points:
        print(f"  {name}: {val:.3f}")
    
    # Check if strictly increasing sequence emerges
    values = [p[0] for p in all_points]
    is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
    
    print(f"\n{'✓' if is_monotonic else '✗'} Global monotonicity: {is_monotonic}")
    
    return all_pass and is_monotonic

if __name__ == "__main__":
    success = test_interleaved_encoding()
    exit(0 if success else 1)
