'''
本代码主要是用HMM模型,实现睡眠错误分期的识别
代码输出结果为某个epoch的预测结果是否正确
0代表神经网络预测结果正确，1代表神经网络预测结果错误
具体实现过程：
以sleepenet网络在train2数据集上的二分类标签作为隐藏状态，结合模型的预测标签
生成HMM模型的参数（π，A，B）
由模型参数和测试机上的预测标签作为输入，反推数据对应的隐藏状态序列
即得到了是否需要修改分期结果的标签
'''
import numpy as np
import os
from collections import defaultdict
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score


# 改变输出框长度，使得矩阵在一行内显示
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=600)  
# 训练数据存储位置（train数据）   
# path_dir = r'C:\Users\31839\Desktop\new_hmm\check'
train_path_dir = r'C:\Users\31839\Desktop\对比实验\Attentionsleep\check\train'
test_path_dir = r'C:\Users\31839\Desktop\对比实验\Attentionsleep\check\test'
# 分段生成隐藏状态
def div_viterbi(predict_label,n,NIU):
    # n暂时取135
    l = len(predict_label)  #序列长度
    h_s_store = list()
    for i in range(0,l,n):
        z = predict_label[i:n+i]
        a = viterbi(z,NIU)
        h_s = a[1][a[0]]
        h_s_store.extend(h_s)
    return h_s_store
# 将隐藏状态转换为数字0，1
def trans_num(seq):
    l = len(seq)
    num = []
    for i in range(l):
        if seq[i] == 'H':
            num.append(0)
        else:
            num.append(1)
    return num
# 将二维字典转换成矩阵形式
def trans_to_arr(dict):
    df = DataFrame(dict).T.fillna(0)
    df = df[[0, 1, 2, 3, 4]]
    emitprob = np.array(df)
    np.set_printoptions(suppress=True)
    return emitprob
# 保存csv文件
def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
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
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))
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
        Pd_data.to_csv('./' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('./' + file_name + '.xls', encoding='utf-8')
#整合二维字典的函数
def sum_value(obj):
    for key in obj:
        if type(obj[key]).__name__ == 'dict':
            if key not in all_state:
                all_state[key] = {}
            for subkey in obj[key]:
                if subkey not in all_state[key]:
                    all_state[key][subkey] = 0
                all_state[key][subkey] += obj[key][subkey]
        else:
            if key not in all_state:
                all_state[key] = 0
            all_state[key] += obj[key]
    return all_state
# 从数据中加载数据，返回真实标签、隐藏状态(预测标签)和修改标签
def load_label(i,dir):
    path = os.path.join(dir,i)
    df = np.load(path)
    tru_label = df['y']
    pre_label = df['z']
    chc_label = df['c']
    chc_label.astype(int)
    label_state = []
    l = len(chc_label)
    for i in range(l):
        if chc_label[i]  == 0:
            label_state.append("H")
        else:
            label_state.append("C")
    return tru_label,pre_label,label_state
# 维特比算法
def viterbi(predict_label,NIU):
    # states_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7}  # 把隐藏状态和索引对应方便后面调用
    states_to_index = NIU.states_to_index
    # index_to_states = ["A", "B", "C", "D", "E", "F", "G", "H"]
    index_to_states = NIU.index_to_states
    states = index_to_states  # 确定隐藏状态
    emit_p = emitprob  # 输入发射矩阵（隐藏状态和观测状态之间的概率对应关系）
    trans_p = transprob # 状态转移矩阵（不同隐藏状态之间的转换概率）
    start_p = startprob # 初始概率矩阵（不同隐藏状态最开始出现的概率）
    #建立临时路径存储变量
    V = [{}]
    #建立最终路径存储变量
    path = {}
    for y in states:
        V[0][y] = start_p[states_to_index[y]] * emit_p[y].get(int(predict_label[0]),0)
        path[y] = [y]
    for t in range(1, len(predict_label)):
        V.append({})
        newpath = {}
        for y in states:
            emitP = emit_p[y].get(int(predict_label[t]), 0)  # 设置未知字单独成词
            temp = []
            for y0 in states:
                if V[t - 1][y0] > 0:  # 踢出概率为零的路径（状态转换）
                    temp.append((V[t - 1][y0] * trans_p[states_to_index[y0], states_to_index[y]] * emitP, y0))
                    # temp.append(1e-10)
                else:
                    temp.append((0,"H"))
            (prob, state) = max(temp)
            # (prob, state) = max([(V[t - 1][y0] * trans_p[states_to_index[y0],states_to_index[y]] * emitP, y0)  for y0 in states if V[t - 1][y0] > 0])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[len(predict_label) - 2][y], y) for y in states])  # 求最大概率的路径
    return state,path
# 定义HMM类
class HMM:
    # 一共两个参数，分别是隐藏状态和修正标签（checklabel）
    def __init__(self,pre_label,check_label):
        # 存储隐藏状态{0，1}分别代表是否需要修正标签
        self.label_state = check_label
        # 存储神经网络预测标签
        self.label = pre_label
        # 对应隐藏状态和索引
        self.states_to_index = {"H":0,"C":1}
        # 定义隐藏状态的索引
        self.index_to_states = ["H","C"]
        # 获取隐藏状态数量（2个）
        self.len_states = len(self.index_to_states)
        # 建立初始概率矩阵，size：1*2
        self.startprob = np.zeros((self.len_states))
        # 建立状态转移矩阵（隐藏状态之间相互转换）size:2*2
        self.transprob = np.zeros((self.len_states, self.len_states))
        # 发射矩阵, 使用的2级字典嵌套
        # 初始化一个total键存储当前状态出现的总次数, 为了后面的归一化使用
        self.emitprob = {"H": {"total": 0},"C": {"total": 0}} 
    # 计算初始概率矩阵
    # 仅有一个参数（隐藏状态）
    def cal_startprob(self,state):
        self.startprob[self.states_to_index[state[0]]] += 1
    # 计算状态转移矩阵
    # 仅有一个参数（隐藏状态）
    def cal_transprob(self,state):
        sta_join = "".join(state)  # 状态转移 从当前状态转移到后一状态, 即 从 sta1 每一元素转移到 sta2 中
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):  # 同时遍历 s1 , s2
            self.transprob[self.states_to_index[s1], self.states_to_index[s2]] += 1
        return self.transprob
    #计算发射矩阵
    def cal_emitprob(self):
        l = len(self.label)
        for i in range(l-1):
            stagediv = self.label[i]
            labelstage = self.label_state[i]
            self.emitprob[labelstage][stagediv] = self.emitprob[labelstage].get(stagediv, 0) + 1
            self.emitprob[labelstage]["total"] += 1
        return self.emitprob 
    # 获取发射矩阵和初始概率矩阵
    def train(self):
        self.cal_emitprob()
        for states in self.label_state:
            self.cal_startprob(states[0])
        return self.startprob,self.emitprob

if __name__ == "__main__":
    # 存储循环中每轮的状态转移矩阵
    trans_store = list()
    # 存储循环中每轮的初始概率矩阵
    start_store = list()
    # 合并存储各轮获取HMM参数的标签
    res0_store = list()
    res1_store = list()
    res2_store = list()
    all_state = defaultdict(defaultdict)
    # 获取文件夹下文件名列表
    train_pathlist = os.listdir(train_path_dir)
    for i in train_pathlist:
        res = load_label(i,train_path_dir)
        # res[1]是网络预测标签，res[2]是checklabel
        NIU = HMM(res[1],res[2])
        trans_prob = NIU.cal_transprob(res[2])
        trans_store.append(trans_prob)
        out = NIU.train()
        start_store.append(out[0])
        emit_prob = sum_value(out[1])
    # 获取2种隐藏状态一共出现的次数，做归一化使用
    sum1,sum2 = emit_prob['H']['total'],emit_prob['C']['total']
    # 删除字典中的无用数据
    del emit_prob['H']['total'];del emit_prob['C']['total']
    # 对数据进行归一化处理，得到的是不同隐藏状态下对应的概率(保留6位小数)
    emit_prob['H'] = {k: round(v / sum1, 6) for k, v in emit_prob['H'].items()}
    emit_prob['C'] = {k: round(v / sum2, 6) for k, v in emit_prob['C'].items()}
    # 对矩阵数据做归一化，获得最终的参数
    startprob = np.around(sum(start_store)/np.sum(sum(start_store)),decimals=5)  #保留5位小数
    transprob = np.around(sum(trans_store) / np.sum(sum(trans_store), axis=1, keepdims=True),decimals=5)    #同上保留五位小数
    emitprob = emit_prob
    emitprob1 = trans_to_arr(emitprob)
    #这里的矩阵中，发射概率矩阵为二维字典
    print("-----基于统计结果的矩阵参数为-----")
    print("初始概率矩阵为：\n", startprob)
    print("状态转移矩阵为：\n", transprob)
    print("发射概率矩阵为：\n", emitprob)
    print("矩阵形式发射概率矩阵为：\n", emitprob1)
    test_pathlist = os.listdir(test_path_dir)
    for i in test_pathlist:
        res = load_label(i,test_path_dir)
        # res1_score 中存储的是所有测试数据的预测标签
        res0_store.extend(res[0])
        res1_store.extend(res[1])
        res2_store.extend(res[2])
    h_s = div_viterbi(res1_store,220,NIU)
    '''
    # 找到最优的序列长度
    acc_store = list()
    pre_store = list()
    recall_store = list()
    for i in range(1,4152):
    # for i in range(1,4152,10):
        h_s = div_viterbi(res1_store,i,NIU)
        a = trans_num(h_s)
        b = trans_num(res2_store)
        acc = accuracy_score(a , b)
        acc_store.append(acc)
        pre = precision_score(a,b)
        pre_store.append(pre)
        recall = recall_score(a,b)
        recall_store.append(recall)
        f1 = f1_score(a,b)
        print(i)
    Data = {'acc': acc_store,'pre': pre_store, 'recall': recall_store}
    Save_to_Csv(data=Data, file_name='demo_find', Save_format='csv', Save_type='col')
    '''

    # print("预测隐藏状态序列：\n",h_s)
    # 预测的checklabel
    a = trans_num(h_s)
    # 真实的checklabel
    b = trans_num(res2_store)
    b_l = len(b)
    suma = sum(a)
    # 第一次判断正确的数量
    count1 = 0
    # 假设第二次判断正确的数量
    count2 = 0
    # print("序列长度为：",b_l)
    # print("序列求和为：",sumb)
    for i in range(b_l):
        if res0_store[i] == res1_store[i]:
            count1 += 1
    for i in range(b_l):
        if res0_store[i] != res1_store[i] and h_s[i] =='C':
            count2 += 1
    
    # 将结果保存成csv文件
    Data = {'true_label': res0_store,'pre_label': res1_store,'true_checklabel': res2_store, 'pre_checklabel': h_s,'truechecklabel_trans':b,'prechecklabel_trans':a}
    Save_to_Csv(data=Data, file_name='demo302', Save_format='csv', Save_type='col')
    acc = accuracy_score(a , b)
    pre = precision_score(a,b)
    recall = recall_score(a,b)
    f1 = f1_score(a,b)
    print("原模型准确率：",accuracy_score(res0_store,res1_store))
    print("所需工作量：",suma/b_l)
    print("机器替代工作量：",1-suma/b_l)
    print("HMM检测准确率为：",acc)
    print("HMM检测精准率为(实际要改的有多少被判定为应该修改)：",pre)
    print("HMM检测召回率为(被判定为应该修改的有多少是实际需要修改的)：",recall)
    print("HMM检测f1分数为：",f1)
    print("假设检测出的异常部分可以全部纠正的准确率：",(count1+count2)/b_l)

