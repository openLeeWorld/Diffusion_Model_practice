import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

x = torch.randn(3, 64, 64) # 더미 데이터
T = 1000
betas = torch.linspace(0.0001, 0.02, T) # 하이퍼 파라미터 생성 (noise scheduling)
# linspace(start, end, steps)

for t in range(T):
    beta = betas[t]
    eps = torch.randn_like(x) # x와 같은 형상의 가우스 노이즈 생성
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps # 노이즈 추가
    
# 이미지 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'flower.png')
image = plt.imread(file_path)
print(image.shape) # (64, 64, 3) (H, W, C)

# 이미지 전처리 정의
preprocess = transforms.ToTensor() 
# PIL이나 numpy.ndarray를 pytorch.tensor로 변환, 
# 픽셀값 범위 min-max scaling, (C, H, W)로 차원 변경
x = preprocess(image)
print(x.shape) # (3, 64, 64)

original_x = x.clone()

def reverse_to_img(x):
    x = x * 255 # 원래 범위로 스케일링 (0, 255)
    x = x.clamp(0, 255) # x의 각 원소를 0~255범위로 제한
    x = x.to(torch.uint8) # torch.uint8로 타입 변환
    to_pil = transforms.ToPILImage() # PIL(PIL 이미지로 변환)
    return to_pil(x)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
imgs = []

for t in range(T):
    if t % 100 == 0: # 노이즈를 100번 추가할 때마다
        img = reverse_to_img(x) # 텐서를 이미지로 복원
        imgs.append(img) # 복원한 이미지 출력
        
    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps
    
# 10개의 이미지를 2행 5열로 표시
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f'Noise: {i * 100}')
    plt.axis('off')

plt.show()

# ============================================
# q(x_t|x_0): 단 한번의 노이즈 추가만으로 확산 과정 구현
# ============================================
def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T
    t_idx = t - 1 # betas[0] is for t=1
    
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0) # 누적곱
    alpha_bar = alpha_bars[t_idx]
    
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t

x = original_x

t = 100
x_t = add_noise(x, t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f'Noise: {t}')
plt.axis('off')
plt.show()
