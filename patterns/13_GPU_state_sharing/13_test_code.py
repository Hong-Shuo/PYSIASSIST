import torch
import time
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

# these are the per channel mean and standart deviation of
# CIFAR10 image database. We will use these to normalize each
# channel to unit deviation with mean 0.

mean_CIFAR10 = np.array([0.49139968, 0.48215841, 0.44653091])
std_CIFAR10 = np.array([0.49139968, 0.48215841, 0.44653091])

# this transformation is used to transform the images to 0 mean and 1 std.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean_CIFAR10, std_CIFAR10)])

# load the CIFAR10 training and test sets
training_set_CIFAR10 = datasets.CIFAR10(root='cifar10/',
                                        transform=transform,
                                        train=True,
                                        download=True)

test_set_CIFAR10 = datasets.CIFAR10(root='cifar10/',
                                    transform=transform,
                                    train=False,
                                    download=True)

print('Number of training examples:', len(training_set_CIFAR10))
print('Number of test examples:', len(test_set_CIFAR10))

# there are ten classes in the CIFAR10 database
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# DataLoaders are used to iterate over the database images in batches rather
# one by one using for loops which is expensive in python since it is interpreted
training_loader_CIFAR10 = torch.utils.data.DataLoader(dataset=training_set_CIFAR10,
                                                      batch_size=512,
                                                      shuffle=True)

test_loader_CIFAR10 = torch.utils.data.DataLoader(dataset=test_set_CIFAR10,
                                                  batch_size=512,
                                                  shuffle=False)


# this function is used to test the accuracy of the model
# over the test set. The network cnn is defined later on in the code.
def test():
    print('Started evaluating test accuracy...')
    cnn.eval()
    # calculate the accuracy of our model over the whole test set in batches
    correct = 0
    for x, y in test_loader_CIFAR10:
        x, y = Variable(x).cuda(), y.cuda()
        h = cnn.forward(x)
        pred = h.data.max(1)[1]
        correct += pred.eq(y).sum()
    return correct / len(test_set_CIFAR10)


# These are the two types of the basic blocks in a residual network. The residual network
# in this code is built by concatenating several such blocks together.
# Basic blocks are of the form x -> D(x) + F(x), where D(x) is x downsampled
# to the same dimensions as F(x) by a single convolution and F(x) is collection of
# successive operations involving several convolutions and batchnorms.
class BasicResBlock1(nn.Module):
    def __init__(self, input, output, downsample, stride=1):
        super(BasicResBlock1, self).__init__()

        self.conv1 = torch.nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm1 = torch.nn.BatchNorm2d(output)
        self.conv2 = torch.nn.Conv2d(output, output, kernel_size=3, padding=1, stride=1, bias=False)
        self.downsample = downsample

        # applied to the residual to downsample

    def forward(self, x1):
        residual = self.downsample(x1)

        x2 = self.conv1(x1)
        x2 = self.batchNorm1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.conv2(x2)

        x2 += residual

        return x2


class BasicResBlock2(nn.Module):
    def __init__(self, input, output):
        super(BasicResBlock2, self).__init__()

        self.conv1 = torch.nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = torch.nn.BatchNorm2d(input)
        self.conv2 = torch.nn.Conv2d(output, output, kernel_size=3, padding=1, stride=1, bias=False)
        self.batchNorm2 = torch.nn.BatchNorm2d(output)
        self.batchNorm3 = torch.nn.BatchNorm2d(output)

    def forward(self, x1):
        residual = x1

        x2 = self.batchNorm1(x1)
        x2 = F.relu(x2, inplace=True)
        x2 = self.conv1(x1);

        x2 = self.batchNorm2(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.conv2(x2)

        x2 += residual

        x2 = self.batchNorm3(x2)
        x2 = F.relu(x2, inplace=True)

        return x2

    # Below we define the residual network class


class ResNet(nn.Module):
    def __init__(self, width, number_of_blocks):
        super(ResNet, self).__init__()

        # these are the inital layers applied before basic blocks

        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = torch.nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace=True)

        # resLayer1 is the basic block for the residual network that is formed by
        # concatenating several basic blocks of increasing dimensions together.
        self.downsample1 = torch.nn.Conv2d(width, 2 * width, kernel_size=1, stride=1, bias=False)
        self.downsample2 = torch.nn.Conv2d(2 * width, 4 * width, kernel_size=1, stride=2, bias=False)
        self.downsample3 = torch.nn.Conv2d(4 * width, 8 * width, kernel_size=1, stride=2, bias=False)

        self.resLayer1 = []
        self.resLayer1.append(BasicResBlock1(width, 2 * width, self.downsample1, 1))
        for x in range(0, number_of_blocks[0]):  # stage1
            self.resLayer1.append(BasicResBlock2(2 * width, 2 * width))
        self.resLayer1 = nn.Sequential(*self.resLayer1)

        self.resLayer2 = []
        self.resLayer2.append(BasicResBlock1(2 * width, 4 * width, self.downsample2, 2))  # stage2
        for x in range(0, number_of_blocks[1]):
            self.resLayer2.append(BasicResBlock2(4 * width, 4 * width))
        self.resLayer2 = nn.Sequential(*self.resLayer2)

        self.resLayer3 = []
        self.resLayer3.append(BasicResBlock1(4 * width, 8 * width, self.downsample3, 2))  # stage3
        for x in range(0, number_of_blocks[2]):
            self.resLayer3.append(BasicResBlock2(8 * width, 8 * width))
        self.resLayer3 = nn.Sequential(*self.resLayer3)

        self.avgpool1 = torch.nn.AvgPool2d(8, stride=1)

        # define the final linear classifier layer
        self.full1 = nn.Linear(8 * width, 10)

        # weight initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant(m.weight, 1)
                torch.nn.init.constant(m.bias, 0)

            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight, mode='fan_out')
                torch.nn.init.constant(m.bias, 0)

    # define the forward run for the input data x
    def forward(self, x):

        # initial layers before basic blocks
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)

        # residual layers and then average pooling
        x = self.resLayer1(x);
        x = self.resLayer2(x);
        x = self.resLayer3(x);
        # x = self.resLayer4(x);

        x = self.avgpool1(x)

        # linear classifier layer (since we
        # use CrossEntropyLoss for the loss function
        # which already has logsoftmax incorporated inside
        # we dont have any activation function here.)
        x = x.view(x.size(0), -1)

        x = self.full1(x)
        return x

    # this is the training function. cnn is the network that is defined later


# optimizer and learning rate lr are modified inside the function

def train(cycles, cost_criterion, cnn, optimizer):
    average_cost = 0  # cost function for the training
    acc = 0  # accuracy over the test set

    for e in range(cycles):  # cycle through the database many times

        print('Cycle: ', e)
        cnn.train()
        loadt = 0
        cudat = 0
        forwardt = 0
        costt = 0
        stept = 0
        avcostt = 0

        # following for loop cycles over the training set in batches
        # of batch_number=5 using the training_loader object

        s1 = time.clock()
        t1 = time.clock()
        for i, (x, y) in enumerate(training_loader_CIFAR10, 0):
            s2 = time.clock()
            loadt = loadt + s2 - s1
            # here x,y will store data from the training set in batches
            x, y = Variable(x).cuda(), Variable(y).cuda()

            s3 = time.clock()
            cudat = cudat + s3 - s2

            h = cnn.forward(x)  # calculate hypothesis over the batch

            s4 = time.clock()
            forwardt = forwardt + s4 - s3

            cost = cost_criterion(h, y)  # calculate cost the cost of the results
            # print(type(cost))
            s5 = time.clock()
            costt = costt + s5 - s4

            optimizer.zero_grad()  # set the gradients to 0
            cost.backward()  # calculate derivatives wrt parameters
            optimizer.step()  # update parameters

            s6 = time.clock()
            stept = stept + s6 - s5

            average_cost += cost.data[0];  # add the cost to the costs

            s1 = time.clock()
            avcostt = avcostt + s1 - s6

        t2 = time.clock()

        print(
            'total time %.2f loading time %.2f, cuda transfer time %.2f, forward time: %.2f, cost time %.2f, step time %.2f, average cost time %.2f' % (
            t2 - t1, loadt, cudat, forwardt, costt, stept, avcostt))
        average_cost = 0


cycles = 50  # number of cycles that the training runs over the database
cost_criterion = torch.nn.CrossEntropyLoss()  # cost function
cnn = ResNet(16, [1, 1, 1]).cuda()  # build the initial network (in the GPU)
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)