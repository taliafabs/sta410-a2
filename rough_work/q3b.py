import numpy as np

# TODO: 3(b) write a function that takes num grid points N as input and returns matrix A and p

# parameters of our problem
L, a, D = 10, 1, 0.1


def get_A_p(N):
    """
    Take the number of grid points, N as an input and return (N-1)x(N-1) matrix A and (N-1) column
    vector p
    """
    # define empty (N-1)x(N-1) matrix A
    A = np.zeros((N-1, N-1))
    # define empty (N-1) column vector p
    p = np.zeros((N-1))
    # use parameter L and input N to define delta x
    delta_x = L/N
    # fill in tridiagonal square matrix A and column vector p
    above_diag = -a/(2*delta_x) + D/(delta_x**2)
    diag = (-2*D)/(delta_x**2)
    below_diag = a / (2 * delta_x) + D / (delta_x ** 2)
    for i in range(N-1):
        x_i = i * delta_x
        A[i][i+1] = above_diag
        A[i][i] = diag
        A[i+1][i] = below_diag
        p[i] = 0.01*(np.cos(2*np.pi*x_i + np.pi)+1) if 0 <= x_i <= 3 else 0
    return A, p


# matrix_A, vector_p = get_A_p(10)
# print(matrix_A)
# print(vector_p)

    # p = np.zeros((N-1))
    # for i in range(N-1):
    #     x_i = i * delta_x
    #     p[i] = 0.01*(np.cos(2*np.pi*x_i + np.pi)+1) if 0 <= x_i <= 3 else 0
    # return A, p

