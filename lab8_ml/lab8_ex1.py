"""Implement L2-norm and L1-norm from scratch"""

import math
def l2_norm(theta_vec):
    return math.sqrt(sum(x**2 for x in theta_vec))

def l1_norm(theta_vec):
    return sum(abs(x) for x in theta_vec)

# Example usage:
theta_vec = [3, 4, -2]
print("L2-norm:", l2_norm(theta_vec))
print("L1-norm:", l1_norm(theta_vec))


