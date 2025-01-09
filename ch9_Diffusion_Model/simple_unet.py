import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential( # 여러 계층을 직렬로 연결
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), # 각 미니배치에서의 평균과 분산을 조정하여 학습을 안정화하고, 학습 속도를 높이는 역할
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.convs(x)
    
class UNet(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        self.down1 = ConvBlock(in_ch, 64)
        self.down2 = ConvBlock(64, 128)
        self.bot1 = ConvBlock(128, 256)
        self.up2 = ConvBlock(128 + 256, 128)
        self.up1 = ConvBlock(64 + 128, 64)
        self.out = nn.Conv2d(64, in_ch, 1)
        
        self.maxpool = nn.MaxPool2d(2) # 맥스 풀링
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') # 업샘플링
        
    def forward(self, x):
        x1 = self.down1(x)
        x = self.maxpool(x1)
        x2 = self.down2(x)
        x = self.maxpool(x2)
        
        x = self.bot1(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1) # tensor(N,C,H,W)에서 인덱스1에 해당하는 채널에 대해 결합
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1) # 스킵 연결
        x = self.up1(x)
        x = self.out(x)
        return x
    
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(10, 1, 28, 28) # 더미 입력
    y = model(x)
    print(y.shape) # (10, 1, 28, 28)
    

            