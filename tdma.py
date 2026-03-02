import numpy as np

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
    b = A[0,0]
    c = A[0,1]
    e[0] = c / b
    f[0] = d[0] / b
    
    # Forward sweep for e
    for i in range(1, n-1):
        a = A[i, i-1]
        b = A[i, i]
        c = A[i, i+1]
        e[i] = c / (b - a*e[i-1])
    
    # Forward sweep for f
    for i in range(1, n):
        a = A[i, i-1]
        b = A[i, i]
        f[i] = (d[i] - a*f[i-1]) / (b - a*e[i-1])
    
    # Back substitution
    x[-1] = f[-1]
    for i in reversed(range(n-1)):
        x[i] = f[i] - e[i]*x[i+1]
    
    return x

