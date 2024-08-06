# Developed by Wang ziyi @DUT
# Theme of work:  数据集准备（测试、训练）定义Mydata类
# Development time:2022/11/24  9:49
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Mydata(Dataset):
    def __init__(self,npz_list):
        super(Mydata, self).__init__()
        x_train = np.load(npz_list[0])["x"]
        y_train = np.load(npz_list[0])["y"]

        #读取所有的npz文件数据，把他们放到一起准备训练
        for npz_file in npz_list[1:]:
            # npz文件中x的格式是（n*3000*1），np.array形式，想把他们加到一起，要用vstack
            x_train = np.vstack((x_train, np.load(npz_file)["x"]))
            # npz文件中y的格式是数组形式的，只需要用append就可以把他们加到一起
            y_train = np.append(y_train, np.load(npz_file)["y"])
            # 此步骤，x_train将所有npz文件中x数据排列在一起，是一个(n*3000*1)的array
        self.len = x_train.shape[0]     #返回x_train中一共有n个30s数据，因为x_train的形状为（n*3000*1）
        self.x_data = torch.from_numpy(x_train)     #将x_train中的数据从numpy转为tensor，存储到x_data中
        self.y_data = torch.from_numpy(y_train).long()      #基本同上

        # 修正输入的输入格式（shape），从（n*3000*1）---->（n*1*3000），使得输入正常
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)  # 使用permute函数交换维度，将x_data变成(n*1*3000)的tensor
        else:       #如果维数不够3维，即增加一维
            self.x_data = self.x_data.unsqueeze(1)
            print(self.x_data.shape)

    #重写方法
    #给出index，返回对应的数据
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        #返回值为torch.tensor形式

    def __len__(self):
        return self.len


if __name__ == '__main__':
    path_dir = r'C:\Users\31839\Desktop\论文代码参考\test_npz'     #给出存储地址
    path_list = os.listdir(path_dir)        #由地址获取文件名,***.npz列表
    count = 0       #计数器（迭代）
    for i in path_list:         #获取不同.npz文件的地址（用path_dir加上文件名列表获得），再重新存入path_list中
        path_list[count] = os.path.join(path_dir, i)
        count += 1
    print(path_list)
    data = Mydata(path_list)        #调用Mydata类，对数据进行处理
    dataloader = DataLoader(data, batch_size=20, shuffle=False)     #shuffle作用是打乱数据并随机排列，这里不可打乱
    for i, (data, label) in enumerate(dataloader):
        print("i:",i)
        print("数据形状\n",data.shape)
        print("标签值\n",label)

