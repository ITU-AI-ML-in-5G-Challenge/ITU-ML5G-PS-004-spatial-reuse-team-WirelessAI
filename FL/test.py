import sys
import csv
sys.path.append(r'C:\ITUchallenge3')
import scipy.io as sio
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import datetime
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from Processing.Generate3 import test_dataset1
#from Processing.Generate3 import test_dataset
#from utils import average_weights
import cnn
import fc
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import random
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
pwd = os.getcwd()+'/'+'result'+ '/'+nowTime
os.makedirs(pwd)

batch_size = 16
learning_rate = 1e-5
local_ep=400
print("pythorch ver is ",torch.__version__)

#test context 序号
st = 0
range_len = 1000
test_len=1000
test_idx=list(range(st, st + range_len))


class subDataset(Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, input1, label1=None, label2=None):
        self.input1 = input1
    #返回数据集大小
    def __len__(self):
        return len(self.input1)
    #得到数据内容和标签
    def __getitem__(self, index):

        input1 = torch.FloatTensor(self.input1[index])
        return input1


global_cnn_model = cnn.CNN()
global_fc_model=fc.FC()
global_cnn_w=global_cnn_model.state_dict()
global_fc_w=global_fc_model.state_dict()
if torch.cuda.is_available():
    print('1')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    global_cnn_model = global_cnn_model.cuda()
    global_fc_model=global_fc_model.cuda()


# 定义损失函数和优化器
#loss_fn = nn.MSELoss(size_average=False)
loss_fn = nn.MSELoss(reduction='sum')

#test
loss3_list = {}
loss4_list = {}
for i in range(test_len):
    loss3_list[i] = []
    loss4_list[i] = []
test_count=0
scen_count=0
test_dataset= test_dataset1()
global_cnn_w=torch.load('C:\ITUchallenge3\\FL\\model\\global_cnn_model\\cnn2.pth')
global_fc_w=torch.load('C:\ITUchallenge3\\FL\\model\\global_fc_model\\fc2.pth')
global_cnn_model.load_state_dict(global_cnn_w)
global_fc_model.load_state_dict(global_fc_w)
countter=0
p=0
total_counter=0
test_out1=[]
test_out2=[]
idx=[]
num=[]
for s in test_idx:
    try:
        print('begin to test senario-', s)

        test_input,counter,p= test_dataset.process_data(s)
        test_data = subDataset(test_input)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        num.append(p)
        idx.append(counter)
        global_cnn_model.eval()
        global_fc_model.eval()
        for j, item in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            input1= item
            input1 = Variable(input1)

            if torch.cuda.is_available():
                input1 = input1.cuda()
            with torch.no_grad():
                out1 = global_cnn_model(input1)
                test_out1.append(out1.cpu().numpy())
                out2 = global_fc_model(out1.data)
                test_out2.append(out2.cpu().numpy())
            test_count += 1

            if test_count == local_ep * 2:
                test_count = 0
        scen_count+=1
    except BaseException as e:
        print(e)

#check test集里多少csv文件
print("csv num is ",idx)

print("througput num is",num)
#print("8A_STA IN",num.index(8,0,len(num)))

#保存test吞吐量预测结果
with open(file="C:\\ITUchallenge3\\FL\\test_result\\final_test_out.csv",mode="w",encoding="gbk") as fp:
    writer=csv.writer(fp)
    writer.writerow(test_idx)
    tout1=np.array(test_out1)
    tout2= np.array(test_out2)
    t_concat= np.hstack((test_out1,test_out2))
    writer.writerows(t_concat.T)


f=open('C:\ITUchallenge3\output_11ax_sr_simulations_test.txt','r+')
flist=f.readlines()
key=1
out2=[]
for s in range(st, st + range_len):
    out2=list(np.round(test_out2[s],2))
    for tp in range(6-num[s]):
        del out2[-1]
    flist[key]=str(out2)+'\n'
    key+=5
    out2=[]
fff=open('C:\ITUchallenge3\output_11ax_sr_simulations_test.txt','w+')
fff.writelines(flist)

f1=open("C:\ITUchallenge3\output_11ax_sr_simulations_test.txt",'r',encoding="gbk")
find_str=["[","]"]
file_line=f1.readlines()
for line in range(len(file_line)):
    if find_str[0] in file_line[line]:
        file_line[line]=file_line[line].replace(find_str[0]," ")
    if find_str[1] in file_line[line]:
        file_line[line] = file_line[line].replace(find_str[1], " ")
f_new=open("C:\ITUchallenge3\output_11ax_sr_simulations_test.txt",'w+',encoding="gbk")
f_new.writelines(file_line)













