import numpy as np
import matplotlib.pyplot as plt
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from common.utils import multivariate_normal

mus = np.array([[2.0, 54.50],
                [4.3, 80.0]]) # 각 가우시안 분포의 평균 벡터
covs = np.array([[[0.07, 0.44],
                [0.44, 33.7]],
                [[0.17, 0.94],
                [0.94, 36.00 ]]]) # 각 가우시안 분포의 공분산 행렬
phis = np.array([0.35, 0.65]) # 각 가우시안 분포의 가중치(뽑힐 확률)

def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    # 각 정규분포의 가중합
    return y

# plot
xs = np.arange(1, 6, 0.1)
ys = np.arange(40, 100, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = gmm(x, phis, mus, covs)
        
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.contour(X, Y, Z)
plt.show()

