"""
Understand the REAL constraints for fuzzy trapezoids.
"""

# Given trapezoids:
# Low =  [4.3,  4.3,  5.0,  5.6]
# Med =  [5.0,  5.3,  5.85, 6.1]
# High = [5.6,  6.1,  6.52, 6.52]

print("Trapezoid validity constraints:")
print("Each trapezoid [a, b, c, d] must satisfy: a ≤ b ≤ c ≤ d")
print()

low = [4.3, 4.3, 5.0, 5.6]
med = [5.0, 5.3, 5.85, 6.1]
high = [5.6, 6.1, 6.52, 6.52]

def check_trapezoid(name, trap):
    print(f"{name}: [{trap[0]:.2f}, {trap[1]:.2f}, {trap[2]:.2f}, {trap[3]:.2f}]")
    checks = [
        (f"{name}[a] ≤ {name}[b]", trap[0] <= trap[1]),
        (f"{name}[b] ≤ {name}[c]", trap[1] <= trap[2]),
        (f"{name}[c] ≤ {name}[d]", trap[2] <= trap[3]),
    ]
    for desc, result in checks:
        print(f"  {'✓' if result else '✗'} {desc}")
    return all(r for _, r in checks)

print("Individual trapezoid validity:")
low_valid = check_trapezoid("Low", low)
print()
med_valid = check_trapezoid("Medium", med)
print()
high_valid = check_trapezoid("High", high)
print()

print("="*60)
print("Interpretability constraint:")
print("Low should be 'leftmost', High should be 'rightmost'")
print()
print("One way to check: compare the 'centers' of each trapezoid")
low_center = (low[1] + low[2]) / 2
med_center = (med[1] + med[2]) / 2
high_center = (high[1] + high[2]) / 2

print(f"Low center:    {low_center:.2f}")
print(f"Medium center: {med_center:.2f}")
print(f"High center:   {high_center:.2f}")
print()
print(f"Low < Medium < High? {low_center < med_center < high_center}")
print()

print("="*60)
print("Do trapezoids overlap? (This is ALLOWED and even desired!)")
print()
print(f"Low ends at {low[3]:.2f}, Medium starts at {med[0]:.2f}")
print(f"  → Overlap: {min(low[3], med[3]) - max(low[0], med[0]):.2f}")
print()
print(f"Medium ends at {med[3]:.2f}, High starts at {high[0]:.2f}")
print(f"  → Overlap: {min(med[3], high[3]) - max(med[0], high[0]):.2f}")
print()

print("="*60)
print("CONCLUSION:")
print("The interleaving is NOT about enforcing a single monotonic sequence!")
print("It's about encoding the parameters in a specific ORDER while ensuring:")
print("  1. Each individual trapezoid is valid (a≤b≤c≤d)")
print("  2. The trapezoids are interpretably ordered (Low < Medium < High overall)")
print("  3. Overlaps between adjacent trapezoids are controlled")
