import numpy as np
import matplotlib.pyplot as plt

mus = np.array([[2.0, 54.50],
                [4.3, 80.0]]) # 각 가우시안 분포의 평균 벡터
covs = np.array([[[0.07, 0.44],
                [0.44, 33.7]],
                [[0.17, 0.94],
                [0.94, 36.00 ]]]) # 각 가우시안 분포의 공분산 행렬
phis = np.array([0.35, 0.65]) # 각 가우시안 분포의 가중치(뽑힐 확률)

def sample():
    k = np.random.choice(2, p=phis) # 가우시안 분포 선택
    mu, cov = mus[k], covs[k]
    x = np.random.multivariate_normal(mu, cov) # 가우시안 분포로부터 샘플링
    return x # x.size = 2

N = 500 # 샘플링 횟수
xs = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample()
    
plt.scatter(xs[:, 0], xs[:, 1], color='orange', alpha=0.7) # 산점도
plt.xlabel('x')
plt.ylabel('y')
plt.show()

