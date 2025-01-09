import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat
    
def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma # 아다므로 곱(원소별 곱)
    return z

class VAE(nn.Module): # 변분형 오토인코더
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum') # 각 원소의 차의 mse합
        L2 = -torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2) # 정규분포의 KL발산에서 비롯 
        return (L1 + L2) / batch_size # 미니 배치에 대한 손실함수의 평균

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    input_dim = 784 # 이미지 데이터x의 크기(28 * 28)
    hidden_dim = 200 # 신경망 중간층의 차원 수
    latent_dim = 20 # 잠재 변수 벡터z의 차원 수
    epochs = 30
    learning_rate = 3e-4
    batch_size = 32
    
    # datasets
    # 사전 학습된 모델을 사용할 경우, 해당 모델이 학습한 데이터셋의 정규화 기준을 따라야 합니다.(일관된 입력 범위를 보장)
    transform = transforms.Compose([ # 여러 변환(transform)을 하나로 묶어 연속적으로 적용
        transforms.ToTensor(), # 텐서로 변환 후, 픽셀 값은 [0, 255] 범위에서 [0, 1]로 정규화(Min-Max 정규화)
        transforms.Lambda(torch.flatten) # 이미지 평탄화 (28, 28) -> (784,)
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # 모델과 옵티마이저
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    # 학습 루프
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        
        for x, label in dataloader: # 데이터로드
            optimizer.zero_grad() # 옵티마이저의 기울기 초기화
            loss = model.get_loss(x) # VAE의 손실 함수 계산
            loss.backward() # 역전파 (기울기 계산)
            optimizer.step() # 매개변수 갱신
            
            loss_sum += loss.item()
            cnt += 1 
            
        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print("current_loss_avg:", loss_avg)
        
    # plot losses
    epochs = list(range(1, epochs+1))
    plt.plot(epochs, losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # generate new images
    with torch.no_grad(): # disables gradient calculation(메모리 절약 효과)
        sample_size = 64
        z = torch.randn(sample_size, latent_dim) # 랜덤 정규분포 데이터 생성
        x = model.decoder(z) # 학습된 디코더에서 생성형 이미지
        generated_images = x.view(sample_size, 1, 28, 28) #  채널(C)이 1인 흑백 이미지(N, C, H, W)
        # view는 연속적인 메모리(layout) 를 가진 텐서에서만 동작
        
    grid_img = torchvision.utils.make_grid(generated_images, nrow=8, padding=2, normalize=True)
    # generated_images를 격자(grid) 형태로 나열하여 하나의 큰 이미지로 생성합니다.
    plt.imshow(grid_img.permute(1, 2, 0)) # 텐서의 차원 순서를 변경 (0,1,2)->(2,1,0)
    plt.axis('off') # 축을 안 보이게 합니다.
    plt.show()
    
