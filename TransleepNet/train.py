import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn,optim
import time
import os
from scipy import signal
from Data_prepare import *
#from newmodel import MMFC
#from multidata_model_twoway2d_lstm import MMCNN
from model import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F

# defination of random seeds for the same initial parameters every times
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# function of transfering list data to csv file
def Save_to_Csv(data, file_name, Save_format = 'csv', Save_type = 'col'):
    # 输入为一个字典，格式： { '列名称': 数据,....} 
    # 列名即为CSV中数据对应的列名， 数据为一个列表
    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储
    # 默认存储在当前路径下
    Name = []
    times = 0
    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1,1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1,1)))
                
            times += 1
            
        Pd_data = pd.DataFrame(columns=Name, data=Data) 
        
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))
        
            times += 1
    
        Pd_data = pd.DataFrame(index=Name, data=Data)  
    
    if Save_format == 'csv':
        Pd_data.to_csv('./'+ file_name +'.csv',encoding='utf-8')
    else:
        Pd_data.to_excel('./'+ file_name +'.xls',encoding='utf-8')
# function of transfering array data to csv file
def trans_array_to_csv(arr,name):
    array = np.array(arr)
    df = pd.DataFrame(array)
    df.to_csv(name+'.csv')
# function of normalization array data
def softmax_rows(arr):
    # 计算每行的指数和
    exp_sums = np.exp(arr).sum(axis=1)
    # 将每行的指数和除以相应的行数，得到每行的归一化因子
    row_sums = exp_sums[:, np.newaxis]
    # 使用归一化因子对每行进行归一化
    normalized_arr = np.exp(arr) / row_sums
    return normalized_arr    
# function of updating npzfiles including data\trulabel\prelabel\checklabel
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
# function of getting npz filename
def npzlist(path):
    npzpathlist = os.listdir(path)
    c = len(npzpathlist)
    for i in range(c):
        npzpathlist[i] = os.path.join(path, npzpathlist[i])
    return npzpathlist
# getting target path
def make_path(path):
    os.makedirs(path+'./check/train') 
    os.makedirs(path+'./check/valid') 
    os.makedirs(path+'./check/test') 
# data loader function, return sth you can use Dataloader directly
def load_data(path):
    #path = r"C:\Users\31839\Desktop\模型尝试\Multimod-with-HMM\测试数据"
    train_path = os.path.join(path,"train")
    valid_path = os.path.join(path,"valid")
    test_path = os.path.join(path,"test")
    trainlist = os.listdir(train_path)
    validlist = os.listdir(valid_path)
    testlist = os.listdir(test_path)
    a = len(trainlist)
    b = len(validlist)
    c = len(testlist)
    for i in range(len(trainlist)):
        trainlist[i] = os.path.join(train_path, trainlist[i])
    for i in range(b):
        validlist[i] = os.path.join(valid_path, validlist[i])
    for i in range(c):
        testlist[i] = os.path.join(test_path, testlist[i])    
    '''
    train_npz_list = npzlist(train_path)
    valid_npz_list = npzlist(valid_path)
    test_npz_list = npzlist(test_path)
    '''
    train = Mydata(trainlist)
    valid = Mydata(validlist)
    test = Mydata(testlist)
    return train,valid,test
def load_data2(path):
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
# defination of train and test function
def trainandtest(data_path):
    #setting parameters for model
    epoch = 300
    print("训练轮数：",epoch)
    Batch_size = 64            
    learning_rate = 0.001
    print("初始学习率为：",learning_rate)
    # saved path for the last pth file
    path_save = r"S/home/ubuntu/Desktop/newmodel/saved_weight/niuben_60.pth"
    all_data = load_data(data_path)
    # getting train dataset
    train_set = all_data[0]
    # getting valid dataset
    valid_set = all_data[1]
    # getting test dataset
    test_set = all_data[2]
    # confirming the devide you used(GPU or CPU)
    device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')        
    # Load data with Dataloader
    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=Batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=Batch_size, shuffle=False)
    # setting random seeds
    same_seeds(42)
    # creating network model(your own model)
    #niuben = MMFC(30)
    niuben = Model()
    print(niuben)
    niuben = niuben.to(device) 
    
    # confirming whether to continue your last model or not     
    if os.path.exists(path_save):
        niuben.load_state_dict(torch.load(path_save))
        print("Loading latest model sucessful!")
    else:
        print("Loading latest model failed!")
    # setting optimizer

    optimizer = torch.optim.Adam(niuben.parameters(), lr=learning_rate, weight_decay=0.0008)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)

    # optimizer = optim.Adam(niuben.parameters(), lr=learning_rate, weight_decay=0.001)
    # setting loss function and pushing it to device
    loss_fn = nn.CrossEntropyLoss()     #crossentropy
    loss_fn = loss_fn.to(device)
    # recording all loss values for mapping
    train_epochs_loss = []
    valid_epochs_loss = []
    test_epochs_loss = []
    # recording acc and other performances(F1-score,Recall,Precision)
    train_acc = []
    valid_acc = []
    test_acc = []
    # path of saving best model(pth file)
    acc_best = 0
    weight_path_best = os.path.join('saved_weight', 'check_point_best.pth')

    for i in range(epoch):
        # time recording
        since = time.time()
        # outs,trgs for making confusion matrix 
        outs_train = np.array([])
        trgs_train = np.array([])
        outs_valid = np.array([])
        trgs_valid = np.array([])
        outs = np.array([])
        trgs = np.array([])
        acc= 0
        nums = 0 

        # ----train process----
        # recording train loss for printing 
        train_epoch_loss = []
        print("--------第{}轮训练开始--------".format(i + 1))
        # Activate it if there exists dropout or batchnorm in your model
        #niuben.train()
        total_train_step = 0  
        for data in train_loader:
            inputs, labels = data
            #inputs = torch.unsqueeze(inputs,1)
            # pushing data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = niuben(inputs)    
            preds = output.data.max(1, keepdim=True)[1].cpu()
            # storing output predicted_label and the true label(target)  
            outs_train = np.append(outs_train, preds.cpu().numpy())  
            trgs_train = np.append(trgs_train, labels.data.cpu().numpy())  
            # activate loss function(crossentropy)
            train_loss = loss_fn(output, labels)
            # activate optimizer
            optimizer.zero_grad()
            # activate backward propagation
            train_loss.backward()
            optimizer.step()
            #step_schedule.step()
            # recording iteration times
            total_train_step = total_train_step + 1        
            # recording train loss
            train_epoch_loss.append(train_loss.item())
            # counting for correct prediction
            a = output.max(axis=1,keepdim = True)[1].T
            b = (a == labels).cpu()
            acc += sum(b[0]).cpu()
            nums += labels.size()[0]        
            # printing outputs every 20 times instead of printing everytime
            if total_train_step % 20 == 0:
                print("训练次数：{}，Loss={}，Lr:{}".format(total_train_step, train_loss.item(),optimizer.state_dict()['param_groups'][0]['lr']))  # 加上item之后可以有助于把tensor类型转换成一个数
        # 可变学习率参数设置
        scheduler.step()
        # recording train_acc for ploting 
        train_acc.append(100 * acc / nums)
        np.save("train_acc.npy", train_acc)
        # recording avg_loss for ploting
        train_epochs_loss.append(np.average(train_epoch_loss))
        # printing acc and train_loss
        print("------第{}轮的训练acc = {:.3f}%, loss = {}-------".format(i+1,100 * acc / nums, np.average(train_epoch_loss)))
        # saving model's parameters every epoch
        if i >100:
            torch.save(niuben.state_dict(), 'saved_weight/niuben_{}.pth'.format(i+1))
            print("--------保存第{}轮模型完成！".format(i+1))
        
        # ----valid process----
        # deactivate backward propogation in valid process
        with torch.no_grad():
            #niuben.eval()
            acc = 0
            nums = 0
            valid_epoch_loss = []
            total_valid_step = 0
            for data in valid_loader:
                inputs, labels = data
                #inputs = torch.unsqueeze(inputs,1)
                # pushing data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = niuben(inputs)    
                preds = output.data.max(1, keepdim=True)[1].cpu()
                # storing output predicted_label and the true label(target)  
                outs_valid = np.append(outs_valid, preds.cpu().numpy())  
                trgs_valid = np.append(trgs_valid, labels.data.cpu().numpy()) 
                
                # recording iteration times
                total_valid_step = total_valid_step + 1  
                # activate loss function(crossentropy)
                valid_loss = loss_fn(output, labels)
                valid_epoch_loss.append(valid_loss.item())
                # counting for correct prediction
                a = output.max(axis=1,keepdim = True)[1].T
                b = (a == labels).cpu()
                acc += sum(b[0]).cpu()
                nums += labels.size()[0]
                if total_valid_step % 10 == 0:
                    print("验证次数：{}，Loss={}".format(total_valid_step, valid_loss.item()))
            
            vacc = 100 * acc / nums
            # saving files for confusion matrix
            np.save("valid_matrix/valid_outs_{}_{}".format(i+1,vacc), outs_valid)
            np.save("valid_matrix/valid_trgs_{}_{}".format(i+1,vacc), trgs_valid)
            # recording train_acc for ploting 
            valid_acc.append(vacc)
            np.save("valid_acc.npy", valid_acc)
            # recording avg_loss for ploting
            valid_epochs_loss.append(np.average(valid_epoch_loss))
            z = 100 * acc / nums
            if epoch > 0 and z >acc_best:
                acc_best = z
                torch.save(niuben.state_dict(), weight_path_best)
                # saving files for confusion matrix
                np.save("valid_outs_best", outs_valid)
                np.save("valid_trgs_best", trgs_valid)
            # printing acc and valid_loss
            print("------第{}轮的验证acc = {:.3f}%, loss = {}-------".format(i+1,z,np.average(valid_epoch_loss)))

        # ----test process----
        # deactivate backward propogation in test process
        with torch.no_grad(): 
            #niuben.eval()
            acc = 0
            nums = 0
            test_epoch_loss = []
            total_test_step = 0  
            for data in test_loader:
                inputs, labels = data
                #inputs = torch.unsqueeze(inputs,1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = niuben(inputs)
                test_loss = loss_fn(output, labels)     
                test_epoch_loss.append(test_loss.item())
                a = output.max(axis=1,keepdim = True)[1].T
                b = (a == labels).cpu()
                acc += sum(b[0]).cpu()
                nums += labels.size()[0]         
                total_test_step = total_test_step + 1       
                preds = output.data.max(1, keepdim=True)[1].cpu()      
                outs = np.append(outs, preds.cpu().numpy())     
                trgs = np.append(trgs, labels.data.cpu().numpy()) 
                if total_test_step % 10 == 0:
                    print("测试次数：{}，Loss={}".format(total_test_step, test_loss.item()))
            
            tacc = 100 * acc / nums
            # saving files for confusion matrix
            np.save("test_matrix/test_outs_{}_{}".format(i+1,tacc), outs)
            np.save("test_matrix/test_trgs_{}_{}".format(i+1,tacc), trgs)  
            test_epochs_loss.append(np.average(test_epoch_loss))
            test_acc.append(100 * acc / nums)
            np.save("test_acc.npy", test_acc)
            print("------第{}轮的测试acc = {:.3f}%, loss = {}-------".format(i+1, 100 * acc / nums,
                                                                        np.average(test_epoch_loss)))
            print("One epoch training completed!")

        time_end = time.time() - since
        print('训练时间： {:.0f}m {:.0f}s'.format(time_end // 60, time_end % 60))

    # getting csv file include train,test,valid acc
    Data = {'trainacc':train_acc, 'validacc':valid_acc, 'testacc':test_acc}
    Save_to_Csv(data = Data, file_name = 'sleepnet_performance', Save_format = 'csv',Save_type = 'col')
# defination of making checklabel function
def make_label(data_path):
    Batch_size = 64
    path_save = r"s/home/ubuntu/Desktop/checknet/niuben_211.pth"
    all_data = load_data2(data_path)
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
    niuben = get_net()
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
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')
    args2 = args.parse_args()
    data_path = str(args2.np_data_dir)
    trainandtest(data_path)
    print("第一阶段实验完成！")
    #make_label(data_path)
    #print("第二阶段实验完成！")







