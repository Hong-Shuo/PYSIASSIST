torch.manual_seed(2809)
modelA = LeNetA()
torch.manual_seed(2809)
modelB = LeNetB()

x = torch.randn(2, 1, 32, 32)
torch.manual_seed(2809)
outputA = modelA(x)
torch.manual_seed(2809)
outputB = modelB(x)