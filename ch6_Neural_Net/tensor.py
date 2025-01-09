import torch

print(torch.__version__)  #2.0.0+cpu

x = torch.tensor(5.0)

y = 3 * x ** 2
print(y)

x = torch.tensor(5.0, requires_grad=True) # 미분 가능하도록 설정
y = 3 * x ** 2

y.backward()
print(x.grad)