import torch
from torch import nn
import math

class DeepSleepnetFeature(nn.Module):
    def __init__(self):
        super(DeepSleepnetFeature,self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.path1 = nn.Sequential(nn.Conv1d(1,64,kernel_size=50,stride=6),
                                   nn.BatchNorm1d(64,eps=1e-5),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=8,stride=8),
                                   nn.Dropout(0.5),
                                   nn.Conv1d(64,128,kernel_size=8,stride=1),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.Conv1d(128,128,kernel_size=8,stride=1),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.Conv1d(128,128,kernel_size=8,stride=1),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4,stride=4),
                                   nn.Flatten()
                                   )
        self.path2 = nn.Sequential(nn.Conv1d(1,64,kernel_size=400,stride=50),
                                   nn.BatchNorm1d(64,eps=1e-5),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=4,stride=4),
                                   nn.Dropout(0.5),
                                   nn.Conv1d(64,128,kernel_size=6,stride=1),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.Conv1d(128,128,kernel_size=6,stride=1),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.Conv1d(128,128,kernel_size=6,stride=1,padding=2),
                                   nn.BatchNorm1d(128,eps=1e-5),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2,stride=2),
                                   nn.Flatten()
                                   )
    def forward(self,x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        # out = self.path2(x)
        out = torch.cat((out1,out2),dim=1)
        return out

class DeepSleepnet(nn.Module):
    def __init__(self):
        super(DeepSleepnet,self).__init__()
        self.extract = DeepSleepnetFeature()
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(1408, 512, num_layers=1,batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(1024, 512, num_layers=1,batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(1408,1024)
        self.fc2 = nn.Linear(2048,5)
    
    def forward(self,x):
        div = self.drop(self.extract(x))
        out1 = self.fc1(div)
        y,_ = self.lstm(div)
        y = self.drop(y)
        out2,_ = self.lstm2(y)
        out2 = self.drop(out2)
        out = torch.cat((out1,out2),dim=1)
        out = self.drop(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    x = torch.randn(5, 1, 3000)        #batch size,1,4*3000
    # x = np.load(r"F:\npz_multidata\SC4001E0.npz")["x"].transpose(0,2,1)
    net = DeepSleepnet()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    y = net(x)
    print(y.shape)
    # print(y[1].shape)
    # print(y[2].shape)