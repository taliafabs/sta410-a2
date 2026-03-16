import numpy as np
import matplotlib.pyplot as plt
import time

# 3(b) write a function that takes num grid points N as input and returns matrix A and p

# parameters of our problem
L, a, D = 10, 1, 0.1


def function_p(x):
    if 0 <= x <= 3:
        return 0.01 * (np.cos(2 * np.pi * x + np.pi) + 1)
    else:
        return 0


def get_A_p(N):
    """
    Take the number of grid points, N as an input and return (N-1)x(N-1) matrix A and (N-1) column
    vector p
    """
    delta_x = L / N
    # build matrix A (I used NumPy documentation)
    main_diag = np.full(N - 1, (-2 * D) / (delta_x ** 2))
    upper_diag = np.full(N - 2, (-a) / (2 * delta_x) + D / (delta_x ** 2))
    lower_diag = np.full(N - 2, a / (2 * delta_x) + D / (delta_x ** 2))
    A = np.diag(main_diag, k=0) + np.diag(upper_diag, k=1) + np.diag(lower_diag, k=-1)
    # build vector p
    # I got this from https://www.programiz.com/python-programming/numpy/vectorization
    x = delta_x * np.arange(1, N)  # x = (1, 2, ..., N-1)^T
    p = np.vectorize(function_p)(x)
    return A, p


# 3(c) use matplotlib.pyplot.spy to visualize the sparsity pattern of A

# N=10 (smaller example for more obvious visualization)
A = get_A_p(10)[0]
plt.spy(A)
m = A.shape[0]
plt.xticks(np.arange(m), np.arange(1, m+1))
plt.yticks(np.arange(m), np.arange(1, m+1))
plt.show()

# N=100
A = get_A_p(100)[0]
plt.spy(A)
m = A.shape[0]
# I used chat gpt to format this
ticks = np.arange(0, m, 10)
plt.xticks(ticks, ticks + 1)
plt.yticks(ticks, ticks + 1)
plt.show()


# 3(d)

# tdma algorithm provided
def solveTDMA(A, d):
    """
    Solve Ax = d for tridiagonal A using Thomas algorithm (TDMA)

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Tridiagonal matrix
    d : ndarray, shape (n,)
        Right-hand side vector

    Returns
    -------
    x : ndarray, shape (n,)
        Solution vector
    """
    A = np.array(A, dtype=float)
    d = np.array(d, dtype=float)

    n = A.shape[0]
    assert A.shape[1] == n, "A is not square"
    assert d.shape[0] == n, "A and d size mismatch"

    # Arrays to store coefficients
    e = np.zeros(n)
    f = np.zeros(n)
    x = np.zeros(n)

    # First row
    b = A[0, 0]
    c = A[0, 1]
    e[0] = c / b
    f[0] = d[0] / b

    # Forward sweep for e
    for i in range(1, n - 1):
        a = A[i, i - 1]
        b = A[i, i]
        c = A[i, i + 1]
        e[i] = c / (b - a * e[i - 1])

    # Forward sweep for f
    for i in range(1, n):
        a = A[i, i - 1]
        b = A[i, i]
        f[i] = (d[i] - a * f[i - 1]) / (b - a * e[i - 1])

    # Back substitution
    x[-1] = f[-1]
    for i in reversed(range(n - 1)):
        x[i] = f[i] - e[i] * x[i + 1]

    return x


# comparing average cost of tdma (solveTDMA) vs lu (np.linalg.solve) vs directly computing
# over 10 runs, using n=1000
# build matrix A and vector p
A, p = get_A_p(1000)
neg_p = -1 * p
total_tdma = 0
total_lu = 0
total_inverting = 0
# run it 10 times to get avg
for i in range(10):
    # tdma
    start = time.time()
    solveTDMA(A, neg_p)
    end = time.time()
    total_tdma += (end - start)
    # lu
    start = time.time()
    np.linalg.solve(A, neg_p)
    end = time.time()
    total_lu += (end - start)
    # directly computing the inverse of A
    start = time.time()
    np.dot(np.linalg.inv(A), neg_p)
    end = time.time()
    total_inverting += (end - start)
average_tdma_runtime = total_tdma/10
average_lu_runtime = total_lu/10
average_cost_inverting = total_inverting/10

print(f"Average runtimes for solving the linear system Au=-p, using N=1000 across 10 runs")
print(f"Using the solveTDMA algorithm: {average_tdma_runtime}")
print(f"Using np.linalg.solve (LU decomposition for general dense matrix): {average_lu_runtime}")
print(f"Directly computing u=A^-1 d, where d=-p: {average_cost_inverting}")


# 3(e)
# at large N from 100 to 1600 how does the cost of each of the solve types
# change as matrix size is doubled? How does the expected cost scaling compare
# to big-N property we discussed in class for each method?

# values of N to try
large_N = [100, 200, 400, 800, 1600]

# solve types: TDMA, np.linalg.solve (LU decomposition)
# keep track of runtimes for each
lu_runtimes = []
tdma_runtimes = []
direct_invert_runtimes=[]

for N in large_N:
    # get matrix A and vector p
    A, p = get_A_p(N)
    neg_p = -1 * p
    tdma_total = 0
    lu_total=0
    direct_invert_total=0
    for i in range(10): # run each solve type 10 times
        # tdma
        start = time.time()
        solveTDMA(A, neg_p)
        end = time.time()
        tdma_total += (end - start)
        # lu
        start = time.time()
        np.linalg.solve(A, neg_p)
        end = time.time()
        lu_total += (end - start)
        # direct (not a good idea!!!!)
        start = time.time()
        np.dot(np.linalg.inv(A), neg_p)
        end = time.time()
        direct_invert_total += (end - start)
    # record average across 10 runs
    tdma_runtimes.append(tdma_total/10)
    lu_runtimes.append(lu_total/10)
    direct_invert_runtimes.append(direct_invert_total/10)

# plot (lu vs tdma)
plt.figure()
plt.plot(large_N, lu_runtimes, label="LU")
plt.plot(large_N, tdma_runtimes, label="TDMA")
plt.xticks(large_N)
plt.xlabel("N")
plt.ylabel("Average runtime (seconds) across 10 runs")
plt.legend()
plt.show()

# plot (all 3)
plt.figure()
plt.plot(large_N, lu_runtimes, label="LU")
plt.plot(large_N, tdma_runtimes, label="TDMA")
plt.plot(large_N, direct_invert_runtimes, label="Computing A^-1 (-p)")
plt.xticks(large_N)
plt.xlabel("N")
plt.ylabel("Average runtime (seconds) across 10 runs")
plt.legend()
plt.show()



