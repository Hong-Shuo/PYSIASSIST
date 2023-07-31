net = ResNet(ResidualBlock, 3, 16, 3, 10).to("cuda")
net.load_state_dict(torch.load("pretrained/ckpt.pth"), strict=False)
net.eval()

with torch.no_grad():
    input = input.to("cuda")
    output = net(input)
    _, predicted = output.max(1)
    print(output)
    print("%5s" % classes[predicted[0]])