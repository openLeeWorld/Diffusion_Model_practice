import torch

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

# x0 = torch.tensor(0.0, requires_grad=True)
# x1 = torch.tensor(2.0, requires_grad=True)

# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

iters = 10000
lr = 0.0015

for i in range(iters):
    if i % 1000 == 0:
        print(x0.item(), x1.item())
        
    y = rosenbrock(x0, x1)
    
    y.backward()
    
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data
    
    # 기울기 재설정(미분값이 누적되므로 초기화)
    x0.grad.zero_()
    x1.grad.zero_()
    
print(x0.item(), x1.item())