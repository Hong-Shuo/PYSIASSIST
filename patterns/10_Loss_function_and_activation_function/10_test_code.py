class conv_Net(nn.Module):
    def __init__(self):
        super(conv_Net,self).__init__()
        self.bn1=nn.BatchNorm2d(3)
        self.Conv1=convNxN(3,64,3)
        self.Conv2=convNxN(64,64,3)
        self.bn2=nn.BatchNorm2d(64)
        self.Conv3=convNxN(64,16,1)
        self.FullC=nn.Linear(in_features=2304,out_features=10)
    def forward(self,x):
        out=activ_Function(self.Conv1(self.bn1(x)))
        out=activ_Function(self.Conv2(out))
        out=F.avg_pool2d(out,(2,2),2,0,False,True,1)
        out=self.bn2(activ_Function(self.Conv3(out)))
        out=F.avg_pool2d(out,(2,2),2,0,False,True,1)
        out=out.view(out.size(0),-1)
        out=self.FullC(out)
        return out

conv_Net()