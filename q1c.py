import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)


def sum_squares_two_pass(x: np.ndarray):
    """Naive two-pass algorithm to compute the sum of squares
    """
    n = x.shape[0]
    # first pass: compute sample mean
    x_bar = np.mean(x)
    total = 0.0
    # second pass
    for i in range(n):
        total += (x[i] - x_bar)**2
    return total


def sum_squares_1b(x: np.ndarray, m: int):
    """
    Compute the sum of squares and correct for error in estimation
    using the algorithm in part (b)

    x: numpy array of size nx1 containing the data (n observations)
    m: size of subset of n used to estimate x_0
    """
    n = x.shape[0]
    if m > n:
        raise ValueError("m must be smaller than n")
    else:
        # compute x_0
        x_subset = np.random.choice(x, size=m, replace=False)
        x_0 = np.mean(x_subset)
        # compute the sum of squares with 1 pass
        sum_x = 0.0
        sum_sq = 0.0
        for i in range(n):
            sum_x += x[i]
            sum_sq += (x[i] - x_0)**2
        x_bar = sum_x / n
        ssb = sum_sq - (n * (x_0 - x_bar)**2)
        return ssb, x_0, x_bar


# draw n=10**3 independent N(10^8, 1) samples
data = stats.norm(loc=1e08, scale=1.).rvs(size=1000)
# data = stats.norm(loc=1e08, scale=1.).rvs(size=100)

# try out different values of m
# m_values = [1, 510, 25, 50, 100, 250, 500]
#
# ssb_10 = []
# x0_10 = []
# for i in range(100):
#     results = sum_squares_1b(x=data, m=1)
#     ssb_10.append(results[0])
#     x0_10.append(results[1])
#
# print(ssb_10)
# print(x0_10)
# print(np.var(ssb_10))

# catastrophic cancelation due to massive roundoff error
# print(data.var())
# 1 pass naive

# forumla from part (b), trying out different m's
ss_two_pass = sum_squares_two_pass(data)
print(f"Sum of squares using two-pass algorithm: {ss_two_pass}")
m_values = [1, 2, 5, 10, 25, 50, 100, 250, 500]
print("Sum of squares using the method defined in 1(b)")
for m in m_values:
    ss_b, x0, xbar = sum_squares_1b(x=data, m=m)
    diff = abs(x0 - xbar)
    diff_ss = abs(ss_b - ss_two_pass)
    print(f"m={m}, ss using 1(b)={ss_b}, x_0={x0}, x_bar={xbar}, x diff={diff}, ss diff={diff_ss}")
