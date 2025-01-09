import numpy as np
import matplotlib.pyplot as plt
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from common.utils import multivariate_normal, gmm

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
original_xs = np.loadtxt(path)

# learned parameters
mus = np.array([[2.0, 54.50],
                [4.3, 80.0]])
covs = np.array([[[0.07, 0.44],
                [0.44, 33.7]],
                [[0.17, 0.94],
                [0.94, 36.00 ]]])
phis = np.array([0.35, 0.65])

# genearte data
N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    k = np.random.choice(2, p=phis)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)

# visualize
plt.scatter(original_xs[:,0], original_xs[:,1], alpha=0.7, label='original')
plt.scatter(new_xs[:,0], new_xs[:,1], alpha=0.7, label='generated')
plt.legend()
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()