import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from module import *
from model import Net
from torchvision import datasets, transforms

# define hyper parameters
Batch_size = 50
Lr = 0.1
Epoch = 1
# define train set and test set
train_dataset = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root='./MNISt',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
# define train loader
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=Batch_size
)
test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.Tensor)
test_y = test_dataset.targets


# print(test_y.shape, test_x.shape)

# construct network
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)  # [200, 64, 28, 28]
        x = self.block2(x)  # [200, 128, 14, 14]
        x = self.block3(x)  # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)  # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [200,256]
        out = self.outlayer(x)
        return out


def my_quantize(model, layer_count, num_bits=8):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            if layer_count[0] == 0:
                quant_conv = QConv2d(child, qi=True, qo=True, num_bits=num_bits)
            else:
                quant_conv = QConv2d(child, qi=False, qo=True, num_bits=num_bits)
            model._modules[name] = quant_conv
        elif isinstance(child, nn.BatchNorm2d):
            quant_bn = QBN(child, qi=False)
            model._modules[name] = quant_bn
        elif isinstance(child, nn.ReLU):
            quant_relu = QReLU()
            model._modules[name] = quant_relu
        elif isinstance(child, nn.MaxPool2d):
            quant_maxpool = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
            model._modules[name] = quant_maxpool
        else:
            my_quantize(child, layer_count, num_bits=num_bits)

def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.forward(data)
        if i % 500 == 0:
            break
    print('direct quantization finish')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True, num_workers=5, pin_memory=True
)
layer_count = [0]
ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)
ResNet18.load_state_dict(torch.load('ckpt/resnet18.pt', map_location='cpu'))
save_file = "ckpt/resnet18_ptq.pt"
my_quantize(ResNet18, layer_count, num_bits=8)
ResNet18.eval()

direct_quantize(ResNet18, train_loader)
ResNet18.forward()
