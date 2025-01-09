import numpy as np

# array 
x = np.array([1, 2, 3])
print(x.__class__) # <class 'numpy.ndarray'>
print(x.shape) # (3,)
print(x.ndim) # 1
W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.ndim) # 2
print(W.shape) # (2, 3)

# element-wise product
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
print(W + X)
print('---')
print(W * X)

# inner product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b) # a @ b
print(y)
print(a @ b)

# matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Y = np.dot(A, B) # A @ B
print(Y)