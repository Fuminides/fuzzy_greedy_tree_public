# Interleaved Fuzzy Partition Encoding

## Overview

This document explains the interpretable encoding scheme used for fuzzy partition optimization, based on the evolutionary algorithm approach.

## The Problem

When optimizing fuzzy partitions (e.g., Low, Medium, High trapezoids), we need to ensure:
1. **Validity**: Each trapezoid [a, b, c, d] satisfies a ≤ b ≤ c ≤ d
2. **Interpretability**: Low < Medium < High (centers are ordered)
3. **Domain Coverage**: Partition spans the entire feature range
4. **Optimizability**: Parameters can be optimized without explicit constraints

## The Solution: Interleaved Encoding

### Key Insight

Instead of storing trapezoid parameters directly, we encode them as **positive increments** in a specific **interleaved order**. During decoding:
1. Accumulate the increments sequentially (guaranteed monotonic)
2. Extract parameters following the interleaved order
3. Normalize to fit the feature domain [feature_min, feature_max]

This ensures validity **by construction** - no constraints needed!

### The Interleaved Order (3 Partitions)

For 3 linguistic terms (Low, Medium, High), the 12 parameters are stored as:

```
Position  1-2:  Low[a, b]      (first two parameters of Low)
Position  3:    Low[c]         (third parameter of Low)
Position  4:    Medium[e]      (first parameter of Medium)
Position  5:    Low[d]         (last parameter of Low)
Position  6:    Medium[f]      (second parameter of Medium)
Position  7:    Medium[g]      (third parameter of Medium)
Position  8:    High[i]        (first parameter of High)
Position  9:    Medium[h]      (last parameter of Medium)
Position  10:   High[j]        (second parameter of High)
Position  11-12: High[k, l]    (last two parameters of High)
```

### Encoding Process

Given trapezoids constructed from quantiles:
```python
Low    = [Q0, Q0, Q1, Q2]
Medium = [Q1, (Q1+Q2)/2, (Q2+Q3)/2, Q3]
High   = [Q2, Q3, Q4, Q4]
```

We encode as **positive increments** between consecutive positions:
```python
enc[0]  = Low[a] - feature_min      # Offset from domain start
enc[1]  = Low[b] - Low[a]           # Increment (0 for flat left)
enc[2]  = Low[c] - Low[b]
enc[3]  = Medium[e] - Low[c]
enc[4]  = Low[d] - Medium[e]
enc[5]  = Medium[f] - Low[d]
enc[6]  = Medium[g] - Medium[f]
enc[7]  = High[i] - Medium[g]
enc[8]  = Medium[h] - High[i]
enc[9]  = High[j] - Medium[h]
enc[10] = High[k] - High[j]
enc[11] = High[l] - High[k]        # Increment (0 for flat right)
```

All encoded values are **≥ 0** (enforced with `max(0, ...)`)

### Decoding Process

1. **Accumulate increments** to create a monotonically increasing sequence:
   ```python
   val[0] = 0
   val[i] = val[i-1] + |enc[i-1]|  # for i = 1 to 12
   ```

2. **Extract parameters** following the interleaved order:
   ```python
   Low[a]    = val[1]
   Low[b]    = val[2]
   Low[c]    = val[3]
   Medium[e] = val[4]
   Low[d]    = val[5]
   Medium[f] = val[6]
   Medium[g] = val[7]
   High[i]   = val[8]
   Medium[h] = val[9]
   High[j]   = val[10]
   High[k]   = val[11]
   High[l]   = val[12]
   ```

3. **Normalize** to feature domain:
   ```python
   # Scale from [Low[a], High[l]] to [feature_min, feature_max]
   scale = (feature_max - feature_min) / (High[l] - Low[a])
   offset = feature_min - Low[a] * scale
   
   # Apply to all parameters
   param_normalized = param * scale + offset
   ```

## Why This Works

### Guarantees

1. **Monotonicity**: Since val[i] = val[i-1] + positive increment, we have:
   ```
   val[1] ≤ val[2] ≤ val[3] ≤ ... ≤ val[12]
   ```

2. **Trapezoid Validity**: The interleaved order ensures:
   ```
   Low[a] ≤ Low[b] ≤ Low[c] ≤ Low[d]     ✓
   Medium[e] ≤ Medium[f] ≤ Medium[g] ≤ Medium[h]  ✓
   High[i] ≤ High[j] ≤ High[k] ≤ High[l]  ✓
   ```
   Each trapezoid's 4 parameters appear in increasing positions in the accumulated sequence.

3. **Interpretability**: After normalization:
   - Low starts at feature_min
   - High ends at feature_max
   - Centers naturally order: Low_center < Medium_center < High_center

4. **Unconstrained Optimization**: Optimizers can modify encoded values freely (only need ≥ 0), and decoded partitions are always valid!

## Example

Starting quantiles: [0, 20, 40, 60, 80, 100] on feature with range [-2.62, 1.85]

**Encoded** (12 positive values):
```
[0.10, 0.00, 1.88, 0.00, 0.43, 0.00, 0.44, 0.00, 0.44, 0.00, 0.53, 0.00]
```

**Accumulated** sequence:
```
[0.00, 0.10, 0.10, 1.98, 1.98, 2.41, 2.41, 2.85, 2.85, 3.29, 3.29, 3.82, 3.82]
```

**Extracted** (before normalization):
```
Low    = [0.10, 0.10, 1.98, 2.41]
Medium = [1.98, 2.41, 2.85, 3.29]
High   = [2.85, 3.29, 3.82, 3.82]
```

**Normalized** to [-2.62, 1.85]:
```
Low    = [-2.62, -2.62, -0.36,  0.16]
Medium = [-0.36,  0.16,  0.69,  1.21]
High   = [ 0.69,  1.21,  1.85,  1.85]
```

Validity checks:
- Each trapezoid valid: ✓
- Centers ordered: -1.49 < 0.42 < 1.53 ✓
- Domain coverage: starts at -2.62, ends at 1.85 ✓

## Generalization to N Partitions

For N linguistic terms, encode 4*N parameters in interleaved order:
- Continue the pattern of mixing parameters from adjacent trapezoids
- All constraints are maintained by the accumulation + normalization approach
- Works for any N ≥ 2

## Benefits

1. **Automatic validity**: No constraint checking needed
2. **Efficient optimization**: Unconstrained positive values
3. **Guaranteed interpretability**: Order preserved by construction
4. **Full coverage**: Normalization ensures domain spanning
5. **Flexible**: Easily extends to N linguistic terms

## Implementation

See `partition_optimization.py`:
- `_encode_partitions_from_quantiles()`: Quantiles → Encoded increments
- `_decode_partitions()`: Encoded increments → Valid trapezoids (normalized)
