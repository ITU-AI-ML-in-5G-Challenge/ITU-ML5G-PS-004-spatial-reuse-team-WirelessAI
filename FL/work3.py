import sys
import csv
sys.path.append(r'C:\ITUchallenge3')
import scipy.io as sio
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import datetime
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from Processing.Generate3 import train_dataset
from Processing.Generate3 import test_dataset
#from utils import average_weights
import cnn
import fc
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import random
import time
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
pwd = os.getcwd()+'/'+'result'+ '/'+nowTime
os.makedirs(pwd)

batch_size = 16
learning_rate = 1e-5
local_ep = 400
global_ep=10
s_num=50
print("pythorch ver is ",torch.__version__)

#train集分成train集和test集,避免重复训练测试同一场景
st = 1000
range_len = 1000
train_len= 15#至少两个,不能超过1000-len(NO_idx)
test_len=20   #不能超过1000-len(NO_idx)-train_len
NO_idx=[1021,1040,1044,1088,1090,1100,1113,1201,1204,1279,1339,1343,1357,1417,1418,1448,1472,1483,1526,1545,1548,1554,1613,1620,1680,1725,1731,1776,1778,1963,1974]#NAN出现的context序号
random_permutation_idx = np.random.permutation(list(range(st, st + range_len)))
list_random_permutation_idx=list(random_permutation_idx)
for k in NO_idx:
    list_random_permutation_idx.remove(k)
train_idx=random.sample(list_random_permutation_idx,train_len)
odd_idx=list_random_permutation_idx
for s in train_idx:
    odd_idx.remove(s)
test_idx=random.sample(list(odd_idx),test_len)

#随机绘制几个context的loss
num_plot=3
random_context=list(np.random.permutation(list(range(0, train_len))))
plot_context=random.sample(random_context,num_plot)




#平均权值
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = torch.load(w[0])
    #w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        tmp=torch.load(w[i])
        for key in w_avg.keys():
            w_avg[key] += tmp[key]
        del tmp
        torch.cuda.empty_cache()

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



class subDataset(Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, input1, label1,label2):
        self.input1 = input1
        self.label1 = label1
        self.label2=label2
    #返回数据集大小
    def __len__(self):
        return len(self.input1)
    #得到数据内容和标签
    def __getitem__(self, index):

        input1 = torch.FloatTensor(self.input1[index])
        label1 = torch.FloatTensor(self.label1[index])
        label2 = torch.FloatTensor(self.label2[index])
        return input1,label1,label2


global_cnn_model = cnn.CNN()
global_fc_model=fc.FC()
global_cnn_w=global_cnn_model.state_dict()
global_fc_w=global_fc_model.state_dict()
if torch.cuda.is_available():
    global_cnn_model = global_cnn_model.cuda()
    global_fc_model=global_fc_model.cuda()
    print(torch.cuda.device_count())

# 定义损失函数和优化器
loss_fn = nn.MSELoss(reduction='mean')


##train
loss1_list=np.zeros((global_ep,train_len,800))
loss2_list=np.zeros((global_ep,train_len,800))
train_count=0
train_dataset = train_dataset()
local_cnn_w=[]
local_fc_w=[]
counter=0
loss11=np.zeros((global_ep,train_len))
loss22=np.zeros((global_ep,train_len))
global_cnn_w1=[]
global_fc_w1=[]
for epoch in  tqdm(range(global_ep)):
    global_cnn_model.train()
    global_fc_model.train()
    scen_count = 0
    print("This global is",epoch)
    for g,s in enumerate(train_idx):
        file_path = "C:\\ITUchallenge3\\train_data_3\\Context" + str(s)
        if os.path.exists(file_path):
            time1 = time.time()
            print('this context is', g)
            train_input, train_label1, train_label2, counter = train_dataset.process_data(s)
            train_data = subDataset(train_input, train_label1, train_label2)
            train_dataloader = DataLoader(train_data, num_workers=0, batch_size=batch_size, shuffle=True)
            cnn_model = copy.deepcopy(global_cnn_model)
            fc_model = copy.deepcopy(global_fc_model)
            print(next(cnn_model.parameters()).device)
            print(time.time() - time1)
            cnn_model.train()
            fc_model.train()
            optimizer1 = optim.Adam(cnn_model.parameters(), lr=learning_rate)
            optimizer2 = optim.Adam(fc_model.parameters(), lr=learning_rate)
            ls1 = []
            ls2 = []
            for i in tqdm(range(local_ep)):

                for j, item in enumerate(train_dataloader):
                    input1, label1, label2 = item
                    input1 = Variable(input1, requires_grad=True)
                    if torch.cuda.is_available():
                        input1 = input1.cuda()
                        label1 = label1.cuda()
                        label2 = label2.cuda()
                    out1 = cnn_model(input1)
                    loss1 = loss_fn(out1, label1)
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()

                    out2 = fc_model(out1.data)
                    loss2 = loss_fn(out2, label2)
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                    train_count += 1

                    if train_count == local_ep * 2:
                        train_count = 0

        else:
            print("this train context haven't exist", s)
            continue



        # 每个本地用户训练完成，保存本地神经网络的权重
        local_cnn_w.append('C:\ITUchallenge3\\FL\\model\\cnn_model\\cnn'+str(g)+'.pth')
        local_fc_w.append('C:\ITUchallenge3\\FL\\model\\fc_model\\fc'+str(g)+'.pth')
        scen_count += 1

        #save model on cpu
        torch.save(cnn_model.state_dict(),'C:\ITUchallenge3\\FL\\model\\cnn_model\\cnn'+str(g)+'.pth')
        torch.save(fc_model.state_dict(), 'C:\ITUchallenge3\\FL\\model\\fc_model\\fc'+str(g)+'.pth')
        del cnn_model
        del fc_model
        torch.cuda.empty_cache()
    #平均网络权重
    global_cnn_w=average_weights(local_cnn_w)
    global_fc_w=average_weights(local_fc_w)
    #更新全局网络的权重
    global_cnn_model.load_state_dict(global_cnn_w)
    global_fc_model.load_state_dict(global_fc_w)
    global_cnn_w1.append(list(global_cnn_w))
    global_fc_w1.append(list(global_fc_w))
    #计算global model 的loss
    for g,s in enumerate(train_idx):
        file_path = "C:\\ITUchallenge3\\train_data_3\\Context" + str(s)
        if os.path.exists(file_path):
            time1 = time.time()
            print('this context is', g)
            train_input, train_label1, train_label2, counter = train_dataset.process_data(s)
            train_data = subDataset(train_input, train_label1, train_label2)
            train_dataloader = DataLoader(train_data, num_workers=0, batch_size=batch_size, shuffle=True)
            cnn_model = global_cnn_model
            fc_model = global_fc_model
            print(next(cnn_model.parameters()).device)
            print(time.time() - time1)
            ls1 = []
            ls2 = []
            for j, item in enumerate(train_dataloader):
                input1, label1, label2 = item
                input1 = Variable(input1, requires_grad=True)
                if torch.cuda.is_available():
                    input1 = input1.cuda()
                    label1 = label1.cuda()
                    label2 = label2.cuda()
                out1 = cnn_model(input1)
                loss1 = loss_fn(out1, label1)
                ls1.append(loss1.data.item())

                out2 = fc_model(out1.data)
                loss2 = loss_fn(out2, label2)
                ls2.append(loss2.data.item())

                train_count += 1

                if train_count == local_ep * 2:
                    train_count = 0
            print("this loss1 is ", np.mean(ls1))
            print("this loss2 is ", np.mean(ls2))
            loss11[epoch][g] = np.mean(ls1)
            loss22[epoch][g] = np.mean(ls2)

        else:
            print("this train context haven't exist", s)
            continue

    #if any(loss11.mean(axis=1) < best_loss1) and any(loss22.mean(axis=1) < best_loss2):
    global_cnn_w1.append('C:\ITUchallenge3\\FL\\model\\global_cnn_model\\cnn'+str(epoch)+'.pth')
    global_fc_w1.append('C:\ITUchallenge3\\FL\\model\\global_fc_model\\fc'+str(epoch)+'.pth')
    torch.save(global_cnn_model.state_dict(),'C:\ITUchallenge3\\FL\\model\\global_cnn_model\\cnn'+str(epoch)+'.pth')
    torch.save(global_fc_model.state_dict(), 'C:\ITUchallenge3\\FL\\model\\global_fc_model\\fc'+str(epoch)+'.pth')

    # 释放显存
    torch.cuda.empty_cache()


loss1_mean = loss11.mean(axis=1)
loss2_mean = loss22.mean(axis=1)
sio.savemat(pwd+'/data.mat', {'train_loss1': loss11,'train_loss2':loss22,'train_loss1_mean':loss1_mean,'train_loss2_mean':loss2_mean})#,'test_loss1': loss3_list,'test_loss3_mean':loss3_mean,'test_loss2':loss4_list,'test_loss4_mean':loss4_mean})

#绘制随机context loss图像
fig=plt.figure()
fig.suptitle('figure:train set of loss')
ax1=fig.add_subplot(321,title='fist context loss1',xlabel='global_ep',ylabel=str(plot_context[0])+"loss1")
ls11=[]
for i in range(global_ep):
    ls11.append(loss11[i][plot_context[0]])
ax1.plot(range(global_ep),ls11,label='LOSS1')
#ax1.savefig(r'C:\ITUchallenge3\FL\log\train_CNN_inter_RSSI_SINR_loss1.png',bbox_inches='tight')

ax2=plt.subplot(322,title='fist context loss2',xlabel='global_ep',ylabel=str(plot_context[0])+"loss2")
ls22=[]
for i in range(global_ep):
    ls22.append(loss22[i][plot_context[0]])
ax2.plot(range(global_ep),ls22,label='LOSS2')
#ax2.savefig(r'C:\ITUchallenge3\FL\log\train_FC_throughput_loss1.png',bbox_inches='tight')

ax3=fig.add_subplot(323,title='second context loss1',xlabel='global_ep',ylabel=str(plot_context[1])+"loss1")
ls11=[]
for i in range(global_ep):
    ls11.append(loss11[i][plot_context[1]])
ax3.plot(range(global_ep),ls11,label='LOSS1')
#ax3.savefig(r'C:\ITUchallenge3\FL\log\train_CNN_inter_RSSI_SINR_loss2.png',bbox_inches='tight')

ax4=plt.subplot(324,title='second context loss2',xlabel='global_ep',ylabel=str(plot_context[1])+"loss2")
ls22=[]
for i in range(global_ep):
    ls22.append(loss22[i][plot_context[1]])
ax4.plot(range(global_ep),ls22,label='LOSS2')


ax5=fig.add_subplot(325,title='third context loss1',xlabel='global_ep',ylabel=str(plot_context[2])+"loss1")
ls11=[]
for i in range(global_ep):
    ls11.append(loss11[i][plot_context[2]])
ax5.plot(range(global_ep),ls11,label='LOSS1')
#ax3.savefig(r'C:\ITUchallenge3\FL\log\train_CNN_inter_RSSI_SINR_loss2.png',bbox_inches='tight')

ax6=plt.subplot(326,title='third context loss2',xlabel='global_ep',ylabel=str(plot_context[2])+"loss2")
ls22=[]
for i in range(global_ep):
    ls22.append(loss22[i][plot_context[2]])
ax6.plot(range(global_ep),ls22,label='LOSS2')
fig.savefig(r'C:\ITUchallenge3\FL\log\train_loss.png',bbox_inches='tight')
plt.show()







