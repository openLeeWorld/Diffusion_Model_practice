import numpy as np
import matplotlib.pyplot as plt


x_means = []
N = 5

for _ in range(100000):
    xs = []
    for i in range(N):
        x = np.random.rand() # 0~1 사이의 난수 생성(균등분포)
        xs.append(x) # 랜덤 샘플링
    mean = np.sum(xs)
    x_means.append(mean)
        
    # normal distribution
def normal(x, mu=0, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

x_norm = np.linspace(-N, N, 100)
mu = 0.5 * N
sigma = np.sqrt(N / 12)
y_norm = normal(x_norm, mu, sigma)

# plot 
plt.hist(x_means, bins='auto', density=True) 
# bins 적당히 조절, density=True: 밀도함수
plt.plot(x_norm, y_norm)
plt.title(f'N={N}')
plt.xlim(-1, 6)
plt.show()