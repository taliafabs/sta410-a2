import numpy as np
from scipy import stats

# define a random number generator
rng = np.random.default_rng(123)

# define sample size n
n=10000

# draw n independent N(10^8, 1) samples
x = stats.norm.rvs(loc=10**8, scale=1, size=n)

# compute the sample mean
x_bar = np.mean(x)

# take x_0 to be the average of a subset of m << n samples
m=100
subset = np.random.choice(x, size=m)
x_0 = np.mean(subset)

# compute the sum of squares and correct for error in estimation
total = 0
for i in range(n):
    total += (x[i] - x_0)**2
ss = total + n*(x_0 - x_bar)**2
print("The sum of squares using the one-pass method in part (b) is:", ss)

