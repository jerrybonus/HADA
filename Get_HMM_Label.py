import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn,optim
import os
import time
from scipy import signal
from Data_prepare import *
from model import SleepNet,CheckNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from torchsummary import summary
import torch.nn.functional as F


# function of getting npz filename
def npzlist(path):
    npzpathlist = os.listdir(path)
    c = len(npzpathlist)
    for i in range(c):
        npzpathlist[i] = os.path.join(path, npzpathlist[i])
    return npzpathlist
# 这里使用mydata进行导入数据（取其中的数据x和标签y）
# 先导入数据制作训练二分类器使用的数据集，获取新标签z
def load_data(path):
    # train_path = os.path.join(path,"train")
    train_path = os.path.join(path,"train2")
    valid_path = os.path.join(path,"valid")
    test_path = os.path.join(path,"test")
    # train_npz_list = npzlist(train_path)
    train_npz_list = npzlist(train_path)
    valid_npz_list = npzlist(valid_path)
    test_npz_list = npzlist(test_path)
    # train = Mydata(train_npz_list)
    train = Mydata(train_npz_list)
    valid = Mydata(valid_npz_list)
    test = Mydata(test_npz_list)
    return train,valid,test
# 将数组转换成csv文件  
def trans_array_to_csv(arr,name):
    array = np.array(arr)
    df = pd.DataFrame(array)
    df.to_csv(name+'.csv')
    
def softmax_rows(arr):
    # 计算每行的指数和
    exp_sums = np.exp(arr).sum(axis=1)
    # 将每行的指数和除以相应的行数，得到每行的归一化因子
    row_sums = exp_sums[:, np.newaxis]
    # 使用归一化因子对每行进行归一化
    normalized_arr = np.exp(arr) / row_sums
    return normalized_arr    
    
# defination of random seeds for the same initial parameters every times
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# 使用函数建立二分类器标签存储位置目标文件夹
def make_path(path):
    os.makedirs(path+'./check/train') 
    os.makedirs(path+'./check/valid') 
    os.makedirs(path+'./check/test') 
def update_npzfile(checklabel,pred,path,oripath):
    '''将新产生的标签写入npz文件,checklabel是使用保存模型在第二部分训练集即path内文件的测试结果
    将输出与真实标签对比得到的，需要将这部分标签重新写如到npz文件内'''
    npzpathlist = os.listdir(path)
    len_npzfile = len(npzpathlist)
    start_point = 0
    for i in range(len_npzfile):
        npzpathlist[i] = os.path.join(path, npzpathlist[i]) # 所有第二部分训练集npz文件路径
        data = np.load(npzpathlist[i])
        # x代表原始数据，y代表真实标签，z代表预测标签,c代表checklabel
        x = data['x']
        y = data['y']
        l = len(y)
        z = pred[start_point:l+start_point]
        c = checklabel[start_point:l+start_point]
        start_point += l
        filename = os.path.basename(npzpathlist[i]) 
        name = "{}".format(filename)
        # oripath是新标签存储位置的空文件夹
        # oripath = r"D:\论文代码参考\test_npz\check"
        targetpath = os.path.join(oripath, name)
        #保存npz文件
        np.savez(targetpath, x=x,y=y,z=z,c=c)
# 使用sleepnet在   第二部分训练数据   上跑测试，与标签对比，生成训练checknet的label（使用这部分的准确率作为权重，重写loss）
# path_save中的路径应该是存储的最优模型参数（在ubuntu上要改路径——linux）
# data_path 是要使用数据集的路径
def make_label(data_path):
    Batch_size = 64
    path_save = r"/home/ubuntu/Desktop/checknet/niuben_211.pth"
    all_data = load_data(data_path)
    outs1 = np.array([])
    outs2 = np.array([])
    outs3 = np.array([])
    # getting train dataset(part two)
    train_set = all_data[0]
    # getting valid dataset
    valid_set = all_data[1]
    # getting test dataset
    test_set = all_data[2]
    # confirming the devide you used(GPU or CPU)
    device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=Batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=False)
    niuben = SleepNet()
    print("睡眠自动分期网络:\n",niuben)
    niuben = niuben.to(device) 
    if os.path.exists(path_save):
        niuben.load_state_dict(torch.load(path_save))
        print("Loading step1 model sucessful!")
    else:
        print("Loading step1 model failed!")
    make_path(data_path)
# getting trainset labels
    with torch.no_grad():
        pred = np.array([])
        label = np.array([])
        acc = 0
        nums = 0
        for data in train_loader:
            inputs, labels = data
            #inputs = torch.unsqueeze(inputs,1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = niuben(inputs) 
            a = output.max(axis=1,keepdim = True)[1].T
            b = (a == labels).cpu()
            acc += sum(b[0]).cpu()
            nums += labels.size()[0] 
            preds = output.data.max(1, keepdim=True)[1].cpu()
            preds = preds.cpu().numpy() 
            labels = labels.data.cpu().numpy()
            pred = np.append(pred,preds)
            label = np.append(label,labels)
        for i in range(len(pred)):
            if pred[i] == label[i]:
                outs1 = np.append(outs1, 0)
            else:
                outs1 = np.append(outs1, 1)   
        # np.save("check_label", outs)            
        print("Getting trainset checklabels completed!")
        #以在第二部分训练数据上的测试准确率作为loss权重
        loss_weight = 100 * acc / nums
        print("模型在第二部分训练集上的准确率为：",loss_weight.item(),"%")
        path1 = os.path.join(data_path, 'train2')
        oripath1 = os.path.join(data_path, 'check')
        oripath1 = os.path.join(oripath1, 'train')
        update_npzfile(outs1,pred,path1,oripath1)
# getting validset labels
    with torch.no_grad():
        pred = np.array([])
        label = np.array([])
        acc = 0
        nums = 0
        for data in valid_loader:
            inputs, labels = data
            #inputs = torch.unsqueeze(inputs,1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = niuben(inputs) 
            a = output.max(axis=1,keepdim = True)[1].T
            b = (a == labels).cpu()
            acc += sum(b[0]).cpu()
            nums += labels.size()[0] 
            preds = output.data.max(1, keepdim=True)[1].cpu()
            preds = preds.cpu().numpy() 
            labels = labels.data.cpu().numpy()
            pred = np.append(pred,preds)
            label = np.append(label,labels)
        for i in range(len(pred)):
            if pred[i] == label[i]:
                outs2 = np.append(outs2, 0)
            else:
                outs2 = np.append(outs2, 1)   
        # np.save("check_label", outs)            
        print("Getting validset checklabels completed!")
        #以在第二部分训练数据上的测试准确率作为loss权重
        valid_acc = 100 * acc / nums
        print("模型在验证集上的准确率为：",valid_acc.item(),"%")
        path2 = os.path.join(data_path, 'valid')
        oripath2 = os.path.join(data_path, 'check')
        oripath2 = os.path.join(oripath2, 'valid')
        update_npzfile(outs2,pred,path2,oripath2)
# getting testset labels
    with torch.no_grad():
        pred = np.array([])
        label = np.array([])
        prob = np.array([])
        acc = 0
        nums = 0
        for data in test_loader:
            inputs, labels = data
            #inputs = torch.unsqueeze(inputs,1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = niuben(inputs) 
            a = output.max(axis=1,keepdim = True)[1].T
            b = (a == labels).cpu()
            acc += sum(b[0]).cpu()
            nums += labels.size()[0] 
            preds = output.data.max(1, keepdim=True)[1].cpu()
            # 输出保存每次分类的五分类概率
            probs = output.data.cpu().numpy()
            prob = np.append(prob,probs)

            preds = preds.cpu().numpy() 
            labels = labels.data.cpu().numpy()
            pred = np.append(pred,preds)
            label = np.append(label,labels)
        label = label.reshape(-1,1)
        pred = pred.reshape(-1,1)
        prob = prob.reshape(-1,5)   
        # 输出值归一化（变成0-1之间的概率） 
        pro = softmax_rows(prob) 
        for i in range(len(pred)):
            if pred[i] == label[i]:
                outs3 = np.append(outs3, 0)
            else:
                outs3 = np.append(outs3, 1)  
        outs3 = outs3.reshape(-1,1) 
        # np.save("check_label", outs)    
        all_inf = np.concatenate((prob,label,pred,outs3),axis=1)
        all_inf01 = np.concatenate((pro,label,pred,outs3),axis=1)
        #all_inf = np.concatenate((all_inf),axis=1)
        title = ['W','N1','N2','N3','REM','tru_label','pre_label','tru_checklabel']
        all_inf = np.vstack((title,all_inf))
        all_inf01 = np.vstack((title,all_inf01))
        print(all_inf.shape)
        trans_array_to_csv(all_inf,'all_information')
        trans_array_to_csv(all_inf01,'all_information_01')

        
                
        print("Getting testset checklabels completed!")
        #以在第二部分训练数据上的测试准确率作为loss权重
        test_acc= 100 * acc / nums
        print("模型测试的准确率为：",test_acc.item(),"%")
        path3 = os.path.join(data_path, 'test')
        oripath3 = os.path.join(data_path, 'check')
        oripath3 = os.path.join(oripath3, 'test')
        update_npzfile(outs3,pred,path3,oripath3)


if __name__ == '__main__':
    '''
    命令行输入的指令
    data_path即数据集路径（初始情况下，路径下包含四个文件夹——train、train2、valid、test）
    '''
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')
    args2 = args.parse_args()
    data_path = str(args2.np_data_dir)
    # data_path = r"D:\论文代码参考\test_npz"      #文件夹内含有三个npz文件
    # data_path 是五分类标签数据存储位置，这里使用makelabel制作二分类标签
    make_label(data_path)
    # 经过上以步骤已经获得了我们训练二分类器需要的数据集（存储在同级目录的check文件夹下）


