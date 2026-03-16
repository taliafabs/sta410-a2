import numpy as np
import matplotlib.pyplot as plt # for 2(d)
rng = np.random.default_rng(123)


# 2(c): Implement Randomized Block Gauss-Seidel

def second_diff_matrix(n):
    """
    Build the (n-2)xn second difference matrix
    """
    M = np.zeros([n-2, n])
    for i in range(n-2):
        M[i][i] =1
        M[i][i+1] = -2
        M[i][i+2] = 1
    return M


def randomized_block_gauss_seidel(input_y, lamb, p, niter):
    """
    Perform randomized-block gauss seidel (question 2c)
    """
    n = input_y.shape[0]
    y = np.concatenate((y_n, np.zeros(n-2)), axis=0)
    M = second_diff_matrix(n)
    X = np.concatenate((np.eye(n), (np.sqrt(lamb)*M)), axis=0)
    # initialize theta hat using vectorized direct solution
    theta_hat = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
    theta_estimates = np.zeros((niter, n))
    objective_values = []
    for k in range(niter):
        # set theta_prev to be theta from previous iteration before computing the new estimate
        theta_prev = theta_hat
        # randomly sample a subset w of size p<n from integers {1, 2, ..., n}
        w = rng.choice(n, size=p, replace=False) # set replace=False so w contains p distinct indices
        c = np.setdiff1d(np.arange(p), w) # get w complement
        # define X_w, X_w_comp, and theta_w_comp
        X_w = X[:, w]
        X_w_comp = X[:, c]
        theta_w_comp = theta_prev[c]
        # minimize the objective function with respect to theta_w
        theta_w_hat = np.linalg.solve(X_w.T @ X_w, X_w.T @ y - X_w.T @ X_w_comp @ theta_w_comp)
        # update theta hat
        theta_hat = theta_prev
        theta_hat[w] = theta_w_hat
        theta_estimates[k] = theta_hat
        loss = np.dot(y - (np.dot(X_w, theta_w_hat) + np.dot(X_w_comp, theta_w_comp)),
                      y - (np.dot(X_w, theta_w_hat) + np.dot(X_w_comp, theta_w_comp))
                      )
        objective_values.append(loss)
    # are we returning value of theta hat after the last iteration or after every iteration?
    return theta_estimates[-1], objective_values


# 2(d)

# load the data
data_array = np.loadtxt("/Users/talia/PycharmProjects/sta410-a2/yield.txt")
y_n = data_array.flatten()

# apply randomized_block_gauss_seidel for different values of p between 5 and 50
values_of_p = [5, 10, 25, 50]

# plot objective vs niter, over 1000 iterations
plt.figure()
for p in values_of_p:
    _, objective = randomized_block_gauss_seidel(input_y=y_n, lamb=2000, p=p, niter=1000)
    plt.plot(objective, label=f"p={p}")

plt.xlabel("Iteration number")
plt.ylabel("Value of objective function")
plt.title("Objective vs Number of Iterations for Values of p Between 5 and 50")
plt.legend()
plt.show()










