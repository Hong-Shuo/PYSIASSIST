def update1(x):
    with torch.no_grad():
        tmp = x - 0.0001*x.grad
    x.grad.zero_()
    return tmp.requires_grad_(True)

def update2(x):
    x.data -=  0.0001*x.grad.data
    x.grad.data.zero_()
    return x

# given A and b, and Ax=b: find x using SGD
A = torch.randn(6, 6, requires_grad=False)
b = torch.randn(6, 1, requires_grad=False)

# choose random x and then search with SGD to find x-hat that is the closest to x in Ax=b
x = torch.randn(6, 1, requires_grad=True)

for i in range(40000):
    loss = torch.norm(A @ x - b)
    loss.backward()
    x = update1(x)
    x = update2(x)
    if not i % 5000: print(f"{loss: >.6f}    {x}")