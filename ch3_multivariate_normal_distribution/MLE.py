import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'height_weight.txt')
xs = np.loadtxt(path)

# Maximum likelihood estimation of multivariate normal distribution
mu = np.mean(xs, axis = 0)  # 평균(데이터 행을 따라서)
cov = np.cov(xs, rowvar = False)  # 공분산(데이터 행을 따라서)

def multivariate_normal(x, mu, cov):
    d = len(x)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    z = 1 / (np.sqrt((2 * np.pi) ** d * det))
    diff = x - mu
    return z * np.exp(-0.5 * np.dot(diff.T, np.dot(inv, diff)))
    # np.dot(diff.T, np.dot(inv, diff)) = diff.T @ inv @ diff

small_xs = xs[:500]
X, Y = np.meshgrid(np.arange(150, 195, 0.5), np.arange(45, 75, 0.5))
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)
        
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis') # cmap: color map
# 3d로 산봉우리 표면 plot을 그림

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(small_xs[:, 0], small_xs[:, 1])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(156, 189)
ax2.set_ylim(36, 79)
ax2.contour(X, Y, Z)
plt.show()
# 등고선 그림을 그림