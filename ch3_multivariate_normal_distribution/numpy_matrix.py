import numpy as np

# transpose
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print('---')
print(A.T)

# inverse matrix
A = np.array([[3, 4], [5, 6]])
invA = np.linalg.inv(A)
print(invA)
print('---')
print("I(elementry matrix): \n", A @ invA)

# determinant
d = np.linalg.det(A)
print(d)

# multi-variate normal distribution
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x-mu).T @ inv @ (x-mu) / -2.0)
    return y

x = np.array([0, 0])
mu = np.array([1, 2])
cov = np.array([[1, 0], [0, 1]])

y = multivariate_normal(x, mu, cov)
print(y)