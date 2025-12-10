"""
Trace through the encoding to understand the pattern correctly.
"""

# Given trapezoids (example values):
# Low = [4.3, 4.3, 5.0, 5.6]
# Med = [5.0, 5.3, 5.85, 6.1]
# High = [5.6, 6.1, 6.52, 6.52]

# Your specification says decoding in order:
# Position 0: Low[a] = 4.3 (absolute)
# Position 1: Low[b] increment from Low[a]  => 4.3 + enc[1] = 4.3, so enc[1] = 0
# Position 2: Low[c] increment from Low[b]  => 4.3 + enc[2] = 5.0, so enc[2] = 0.7
# Position 3: Medium[e] increment from Low[c] => 5.0 + enc[3] = 5.0, so enc[3] = 0
# Position 4: Low[d] increment from Medium[e] => 5.0 + enc[4] = 5.6, so enc[4] = 0.6
# Position 5: Medium[f] increment from Low[d] => 5.6 + enc[5] = 5.3, so enc[5] = -0.3 ❌ NEGATIVE!

# Ah! This reveals the issue. The cumulative sum must go FORWARD, but the trapezoid
# parameters themselves don't follow a simple left-to-right order!

# Let me re-think...  The key must be that we're tracking MULTIPLE cumulative sums
# Or the ordering you gave means something different about which previous value to add to.

print("Tracing the pattern from the specification...")
print()
print("Given trapezoids:")
print("Low =  [4.3,  4.3,  5.0,  5.6]   (a, b, c, d)")
print("Med =  [5.0,  5.3,  5.85, 6.1]   (e, f, g, h)")
print("High = [5.6,  6.1,  6.52, 6.52]  (i, j, k, l)")
print()

# Mapping according to the user's specification:
# 1-2: Low[a,b]
# 3: Low[c]
# 4: Medium[e]
# 5: Low[d]
# 6: Medium[f]
# 7: Medium[g]
# 8: High[i]
# 9: Medium[h]
# 10: High[j]
# 11-12: High[k,l]

# So the sequence of VALUES in encoded order is:
values_in_order = [
    4.3,   # 1: Low[a]
    4.3,   # 2: Low[b]
    5.0,   # 3: Low[c]
    5.0,   # 4: Medium[e]
    5.6,   # 5: Low[d]
    5.3,   # 6: Medium[f]
    5.85,  # 7: Medium[g]
    5.6,   # 8: High[i]
    6.1,   # 9: Medium[h]
    6.1,   # 10: High[j]
    6.52,  # 11: High[k]
    6.52,  # 12: High[l]
]

print("Values in encoding order:")
for i, v in enumerate(values_in_order, 1):
    print(f"  Position {i:2d}: {v:.2f}")
print()

# Check monotonicity of this sequence
is_monotonic = all(values_in_order[i] <= values_in_order[i+1] for i in range(len(values_in_order)-1))
print(f"Is this sequence monotonically increasing? {is_monotonic}")
print()

# If we encode as simple cumulative increments:
enc = [values_in_order[0]]  # First value absolute
for i in range(1, len(values_in_order)):
    enc.append(values_in_order[i] - values_in_order[i-1])

print("Incremental encoding:")
for i, e in enumerate(enc):
    print(f"  enc[{i:2d}] = {e:6.2f}")
print()

# Check: some are negative!
negatives = [(i, e) for i, e in enumerate(enc) if e < 0]
if negatives:
    print("WARNING: Negative increments found:")
    for i, e in enumerate(enc):
        if e < 0:
            print(f"  enc[{i}] = {e:.2f}  (goes from {values_in_order[i-1]:.2f} to {values_in_order[i]:.2f})")
else:
    print("✓ All increments are non-negative!")
