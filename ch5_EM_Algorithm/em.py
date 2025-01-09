# EM(Expectation Maximization) 알고리즘을 구현
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from common.utils import multivariate_normal, gmm

path = os.path.join(os.path.dirname(__file__), 'old_faithful.txt')
xs = np.loadtxt(path)
print(xs.shape) # (272, 2)

# initialize parameters
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)]) # np.eye로 단위 행렬 생성

K = len(phis) # 가우스 분포 개수 2
N = len(xs) # 데이터 수 272
MAX_ITERS = 100 # EM알고리즘의 최대 반복 횟수
THRESHOLD = 1e-4 # EM알고리즘의 갱신 중지 임곗값

def likelihood(xs, phis, mus, covs): 
    # log likelihood
    eps = 1e-8 # log(0)방지용 미세한 값 더하기
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N

current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS): # E-M STEP 반복 루프
    # E-STEP ======================================================
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)
        
    # M-step ===========================================
    qs_sum = qs.sum(axis=0) # n 따라 합
    for k in range(K):
        # 1. phis
        phis[k] = qs_sum[k] / N
        
        # 2. mus
        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]      
        
        # 3. covs
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis] # column vector
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]
    # threshold check ===================================
    print(f'{current_likelihood:.3f}') # 로그 가능도 출력(소수점 이하 3자리)
    
    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood)
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood
    
# visualize 
def plot_contour(w, mus, covs): # weights, 평균 벡터, 공분산 벡터
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            
            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)
    
plt.scatter(xs[:, 0], xs[:, 1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(min)')
plt.ylabel('Waiting(Min)')
plt.show()

