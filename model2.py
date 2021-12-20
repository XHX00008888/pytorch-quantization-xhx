from module import *

class MyResNet18(nn.Module):
    def __init__(self, num_channels=1):
        super(MyResNet18, self).__init__()

        # struct: conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # struct: maxpool
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # struct: block1
        self.conv2 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        # struct: block2
        self.conv6 = nn.Conv2d(16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(32)
        self.bn9 = nn.BatchNorm2d(32)

        self.conv1x1_0 = nn.Conv2d(16, out_channels=32, kernel_size=1, stride=2)
        # struct: block3
        self.conv10 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv11 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        self.bn13 = nn.BatchNorm2d(64)

        self.conv1x1_1 = nn.Conv2d(32, out_channels=64, kernel_size=1, stride=2)
        # struct: avg
        self.avgpool = nn.AvgPool2d(7)
        # struct: Linear
        self.outlayer = nn.Linear(64, 10)
    def forward(self,x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.maxpool1(out)
        # struct: block1
        out = self.conv2(out1)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out1 = self.relu(out + out1)

        out = self.conv4(out1)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out1 = self.relu(out + out1)
        # struct: block2
        out = self.conv6(out1)
        out = self.bn6(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out1 = self.conv1x1_0(out1)
        out = self.relu(out + out1)

        out = self.conv8(out)
        out = self.bn8(out)
        out = self.relu(out)

        out = self.conv9(out)
        out = self.bn9(out)
        out1 = self.relu(out + out1)
        # struct: block3
        out = self.conv10(out1)
        out = self.bn10(out)
        out = self.relu(out)

        out = self.conv11(out)
        out = self.bn11(out)
        out1 = self.conv1x1_1(out1)
        out = self.relu(out + out1)

        out = self.conv12(out)
        out = self.bn12(out)
        out = self.relu(out)

        out = self.conv13(out)
        out = self.bn13(out)
        out = self.relu(out + out1)
        # struct: avg
        out = self.avgpool(out)
        # struct: Linear
        out = out.view(out.size(0), -1)
        out = self.outlayer(out)

        return out
    def quantize(self, num_bits=8):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=3, stride=1, padding=1)
        # struct: block1
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qconv3 = QConvBN(self.conv3, self.bn3, qi=False, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU(qi=True)
        self.qconv4 = QConvBNReLU(self.conv4, self.bn4, qi=False, qo=True, num_bits=num_bits)
        self.qconv5 = QConvBN(self.conv5, self.bn5, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU(qi=True)
        # struct: block2
        self.qconv6 = QConvBNReLU(self.conv6, self.bn6, qi=False, qo=True, num_bits=num_bits)
        self.qconv7 = QConvBN(self.conv7, self.bn7, qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU(qi=True)
        self.qconv8 = QConvBNReLU(self.conv8, self.bn8, qi=False, qo=True, num_bits=num_bits)
        self.qconv9 = QConvBN(self.conv9, self.bn9, qi=False, qo=True, num_bits=num_bits)
        self.qrelu4 = QReLU(qi=True)
        # struct: block3
        self.qconv10 = QConvBNReLU(self.conv10, self.bn10, qi=False, qo=True, num_bits=num_bits)
        self.qconv11 = QConvBN(self.conv11, self.bn11, qi=False, qo=True, num_bits=num_bits)
        self.qrelu5 = QReLU(qi=True)
        self.qconv12 = QConvBNReLU(self.conv12, self.bn12, qi=False, qo=True, num_bits=num_bits)
        self.qconv13 = QConvBN(self.conv13, self.bn13, qi=False, qo=True, num_bits=num_bits)
        self.qrelu6 = QReLU(qi=True)
        self.qconv1x1_0 = QConv2d(self.conv1x1_0, qi=False, qo=True)
        self.qconv1x1_1 = QConv2d(self.conv1x1_1, qi=False, qo=True)
        self.qavg = QAvgPooling2d(7)
        self.qfc = QLinear(self.outlayer, qi=False, qo=True, num_bits=num_bits)
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x1 = self.qmaxpool2d_1(x)
        # struct: block1
        x = self.qconv2(x1)
        x = self.qconv3(x)
        x1 = self.qrelu1(x + x1)

        x = self.qconv4(x1)
        x = self.qconv5(x)
        x1 = self.qrelu2(x + x1)
        # struct: block2
        x = self.qconv6(x1)
        x = self.qconv7(x)
        x1 = self.qconv1x1_0(x1)
        x1 = self.qrelu3(x + x1)

        x = self.qconv8(x1)
        x = self.qconv9(x)
        x1 = self.qrelu4(x + x1)
        # struct: block3
        x = self.qconv10(x1)
        x = self.qconv11(x)
        x1 = self.qconv1x1_1(x1)
        x1 = self.qrelu5(x + x1)

        x = self.qconv12(x1)
        x = self.qconv13(x)
        x1 = self.qrelu6(x + x1)

        x = self.qavg(x1)
        out = x.view(x.size(0), -1)
        out = self.qfc(out)
        return out
    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        # struct: block1
        self.qconv2.freeze(self.qconv1.qo)
        self.qconv3.freeze(self.qconv2.qo)
        self.qrelu1.freeze()
        self.qconv4.freeze(self.qrelu1.qi)
        self.qconv5.freeze(self.qconv4.qo)
        self.qrelu2.freeze()
        # struct: block2
        self.qconv6.freeze(self.qrelu2.qi)
        self.qconv7.freeze(self.qconv6.qo)
        self.qconv1x1_0.freeze(self.qrelu2.qi)
        self.qrelu3.freeze()
        self.qconv8.freeze(self.qrelu3.qi)
        self.qconv9.freeze(self.qconv8.qo)
        self.qrelu4.freeze()
        # struct: block3
        self.qconv10.freeze(self.qrelu4.qi)
        self.qconv11.freeze(self.qconv10.qo)
        self.qconv1x1_1.freeze(self.qrelu4.qi)
        self.qrelu5.freeze()
        self.qconv12.freeze(self.qrelu5.qi)
        self.qconv13.freeze(self.qconv12.qo)
        self.qrelu6.freeze()

        self.qavg.freeze(self.qrelu6.qi)
        self.qfc.freeze(self.qrelu6.qi)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx1 = self.qmaxpool2d_1.quantize_inference(qx)
        # struct: block1
        qx = self.qconv2.quantize_inference(qx1)
        qx = self.qconv3.quantize_inference(qx)
        qx1 = self.qrelu1.quantize_inference(qx + qx1)

        qx = self.qconv4.quantize_inference(qx1)
        qx = self.qconv5.quantize_inference(qx)
        qx1 = self.qrelu2.quantize_inference(qx + qx1)
        # struct: block2
        qx = self.qconv6.quantize_inference(qx1)
        qx = self.qconv7.quantize_inference(qx)
        qx1 = self.qconv1x1_0.quantize_inference(qx1)
        qx1 = self.qrelu3.quantize_inference(qx + qx1)

        qx = self.qconv8.quantize_inference(qx1)
        qx = self.qconv9.quantize_inference(qx)
        qx1 = self.qrelu4.quantize_inference(qx + qx1)
        # struct: block3
        qx = self.qconv10.quantize_inference(qx1)
        qx = self.qconv11.quantize_inference(qx)
        qx1 = self.qconv1x1_1.quantize_inference(qx1)
        qx1 = self.qrelu5.quantize_inference(qx + qx1)

        qx = self.qconv12.quantize_inference(qx1)
        qx = self.qconv13.quantize_inference(qx)
        qx1 = self.qrelu6.quantize_inference(qx + qx1)

        qx = self.qavg.quantize_inference(qx1)
        out = qx.view(qx.size(0), -1)
        out = self.qfc.quantize_inference(out)
        out = self.qfc.qo.dequantize_tensor(out)

        return out










