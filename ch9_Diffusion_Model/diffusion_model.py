import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm # 진행률 표시줄을 출력하는 라이브러리

# 시각 데이터 하나(정수)를 인코딩하는 함수
def _pos_encoding(t, output_dim, device='cpu'): # 'cuda'입력시 GPU 사용
    D = output_dim
    v = torch.zeros(D, device=device)
    
    i = torch.arange(0, D, device=device) # [0, 1, .. D-1, D]
    div_term = 10000 ** (i / D)
    
    v[0::2] = torch.sin(t / div_term[0::2]) # 슬라이스 기법을 통해 짝수만 적용
    v[1::2] = torch.cos(t / div_term[1::2]) # 슬라이스 기법을 통해 홀수만 적용
    return v

# 배치 데이터를 처리하는 사인파 위치 인코딩
def pos_encoding(ts, output_dim, device='cpu'):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
        # 텐서의 각 원소에 대해 사인파 위치 인코딩 호출
    return v

class ConvBlock(nn.Module): # UNET의 모듈용 컨볼루션 블록
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential( # 완전 연결 계층
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        ) # (N, D) -> (N, C)
        
    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v) # (N, D) -> (N, C)
        v = v.view(N, C, 1, 1) # RESHAPE
        y = self.convs(x + v) # broadcast(각 pixel에 위치 정보 더함)
        return y
    
class UNet(nn.Module): # 노이즈 예측 모델 
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x, timesteps):
        # 사인파 위치 인코딩
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)
        # x.device: 텐서 x가 어떤 디바이스에 저장되어 있는가?
        x1 = self.down1(x, v) # 신경망에 v도 입력
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        
        x = self.bot1(x, v)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x
    
class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                device='cpu'):
        self.num_timesteps= num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps,
                                    device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t): # 확산 과정
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # t의 모든 원소가 1~T 범위인지 확인
        t_idx = t - 1
        
        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0) # 첫번째 차원의 크기: 배치 크기
        alpha_bar = alpha_bar.view(N, 1, 1, 1) # alpha_bar의 형상 변환
        
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    
    def denoise(self, model, x, t): # 역확산 과정
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all() # t의 모든 원소가 1~T 범위인지 확인
        t_idx = t - 1
        
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]
        
        # 브로드캐스트가 올바르게 수행되도록 설정
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)
        
        model.eval() # 평가모드로 전환(배치 정규화와 드롭아웃 등 학습 때와 달라짐)
        with torch.no_grad(): # 기울기 계산 비활성화(메모리 사용량 감소)
            eps = model(x, t) # 노이즈 예측
        model.train() # 학습 모드로 전환
        
        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0 # no noise at t = 1
        """
        t == 1에 의해 t의 각 원소가 1이면 True, 그렇지 않으면 False인 불리언 타입 텐서
        True인 원소를 noise에서 선택함.
        선택된 원소에 새로운 값 0을 할당(원래 값은 1)
        """
        
        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std
        
    def reverse_to_img(self,x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)
        
    def sample(self, model, x_shape=(20, 1, 28, 28)): # 이미지 생성 (배치 수 20)
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        
        for i in tqdm(range(self.num_timesteps, 0, -1)): # 역순으로 순환
            # tqdm()에 역순 반복자를 전달해 반복문 진행상황을 표시하는 진행룰 표시줄 만듬
            t = torch.tensor([i] * batch_size, device=self.device,
                            dtype=torch.long) 
            
            x = self.denoise(model, x, t) # 노이즈 제거
            
        # 각 배치의 텐서들을 이미지로 변환
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images
    
def show_images(images, rows=2, cols=10): # 확산모델이 생성한 이미지 표시
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()
    
if __name__ == "__main__":
    # hyperparameter 정의
    img_size = 28
    batch_size = 128
    num_timesteps = 1000
    epochs = 10
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 확산 모델 학습용 데이터 준비
    preprocess = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='/data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 확산 모델 객체 준비
    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet() 
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 신경망 학습(loss 최소화)
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        
        # 에포크마다 데이터를 생성하여 결과를 확인(주석 가능)
        # images = diffuser.sample(model)
        # show_images(images)
        
        for images, labels in tqdm(dataloader): # 진행률 표시줄 사용
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps+1, (len(x),), device=device)
            
            x_noisy, noise = diffuser.add_noise(x, t) # 시각 t의 노이즈 이미지 생성 
            noise_pred = model(x_noisy, t)  # 모델로 노이즈 예측
            loss = F.mse_loss(noise, noise_pred) # 실제 노이즈와의 제곱 오차 계산
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            cnt += 1
            
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')
        
    # 손실 그래프
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # 학습된 모델로 이미지 생성
    images = diffuser.sample(model)
    show_images(images)