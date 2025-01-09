import torch
import torch.nn.functional as F

torch.manual_seed(0) # 시드 고정
x = torch.rand(100, 1) # 랜덤 수 생성
y = 2 * x + 5 + torch.rand(100, 1)

torch.manual_seed(0)
x = torch.rand(100, 1)
y = 5 + 2 * x + torch.rand(100, 1)

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    y = x @ W + b  # @는 파이썬 행렬 곱셈 연산자로, __matmul__를 구현하면 됨
    return y # (100, 1)

def mean_squared_error(x0, x1): # 평균 제곱 오차
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

lr = 0.1
iters = 100

for i in range(iters):
    y_hat = predict(x)
    #loss = mean_squared_error(y, y_hat)
    loss = F.mse_loss(y, y_hat)
    
    loss.backward()
    
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data  #optimizer.step() 과정
    
    W.grad.zero_()
    b.grad.zero_()
    
    if i % 10 == 0:
        print(loss.item())
        
print(loss.item())

print('===')
print('W=', W.item())
print('b=', b.item())
    
    

