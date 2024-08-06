import torch
import torch.nn as nn


def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFeature(nn.Module):

    def __init__(self):

        super(ResNetFeature, self).__init__()

        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        # self.config = config
        self.layers = self.layer_config_dict[50]
        block = Bottleneck
        
        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.dropout = nn.Dropout(p=0.5)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, first=False):

        downsample = None
        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f_seq = []

        for i in range(1):
            f = self.initial_layer(x[:, i].view(x.size(0), 1, -1))
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.maxpool(f)
            f = self.layer3(f)
            f = self.layer4(f)
            f = self.dropout(f)
            f_seq.append(f.permute(0, 2, 1))

        out = torch.cat(f_seq, dim=1)

        return out


class PlainLSTM(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(PlainLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_classes = num_classes

        self.input_dim = 128

        # architecture
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def init_hidden(self, x):
        h0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_dim)).cuda()
        c0 = torch.zeros((self.num_layers * 2, x.size(0), self.hidden_dim)).cuda()
        
        return h0, c0

    def forward(self, x):
        hidden = self.init_hidden(x)
        out, hidden = self.lstm(x, hidden)

        out_f = out[:, -1, :self.hidden_dim]
        out_b = out[:, 0, self.hidden_dim:]
        out = torch.cat((out_f, out_b), dim=1)
        out = self.fc(out)

        return out


class IITNET(nn.Module):
    
    def __init__(self):

        super(IITNET, self).__init__()

        self.feature = ResNetFeature()
        self.classifier = PlainLSTM( hidden_dim=128, num_classes=5)

    def forward(self, x):
        out = self.classifier(self.feature(x))

        return out

if __name__ == '__main__':
    x = torch.randn(5, 1, 3000)        #batch size,1,4*3000
    # x = np.load(r"F:\npz_multidata\SC4001E0.npz")["x"].transpose(0,2,1)
    net = IITNET()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    y = net(x)
    print(y.shape)
    # print(y[1].shape)
    # print(y[2].shape)