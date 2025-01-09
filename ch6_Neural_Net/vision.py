# torchvision: 파이토치용 이미지 처리 라이브러리(데이터셋 로딩, 이미지 전처리 등)
import torch
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

print(torchvision.__version__) # 0.15.1+cpu

dataset = torchvision.datasets.MNIST(
    root='./data', # 데이터셋을 저장할 디렉터리를 지정
    train=True, # 학습용 데이터셋, False면 테스트용 데이터셋
    transform=None, # 이미지 테이터 적용할 전처리 
    download=True # 데이터셋이 겨오레 없을 시 자동으로 다운로드 한다.
)

# 데이터셋에서 0번째 이미지 선택
x, label = dataset[0]

print('size: ', len(dataset)) # 60000
print('type: ', type(x)) # PIL.Image.Image
# Image클래스를 pytorch에서 사용하려면 Tensor클래스로 변환해야 함.
print('label: ', label) # 5

# 이미지 출력
plt.imshow(x, cmap='gray')
plt.show()

# === preprocess
transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root='./data', # 데이터셋을 저장할 디렉터리를 지정
    train=True, # 학습용 데이터셋, False면 테스트용 데이터셋
    transform=transform, # 이미지 테이터 적용할 전처리 (텐서로 전환)
    download=True # 데이터셋이 겨오레 없을 시 자동으로 다운로드 한다.
)

x, label = dataset[0]

print('type: ', type(x)) # torch.Tensor
print('shape: ', x.shape) # torch.Size([1, 28, 28])

# === DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for x, label in dataloader:
    print('x shape:', x.shape)
    print('label shape:', label.shape)
    break # 0번째 미니배치 정보만 표시하고 루프 빠져나감