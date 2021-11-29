import numpy as np
import pandas as pd
import os


def mkdir(path):
    # 引入模块

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False



find='KOMONDOR'


filename = 'output_11ax_sr_simulations_sce2.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
#filename='test.txt'
pos = []
count1=0
count2=0
s=3000
with open(filename, 'r',errors='ignore') as file_to_read:
   while True:
     lines = file_to_read.readline() # 整行读取数据
     if not lines:
        break
        pass
     if (find in lines):
         a = lines.split("\'")[1]
         input_file_name=a.split(".csv")[0]+'_output'
     else:
         filter(str.isdigit, lines)#每一行数据可能有乱码，需要滤除
         p_tmp = [float(i) for i in lines.split(',')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
         pos.append(p_tmp)  # 添加新读取的数据
     count1+=1
     if count1==5:
         if count2>=0 and count2!=21:
             count2+=1
         thr=[str(i) for i in pos[0]]
         thr=','.join(thr)
         rssi=[str(i) for i in pos[2]]
         rssi=','.join(rssi)
         rssi = '\'' + rssi
         sinr=[str(i) for i in pos[3]]
         sinr=','.join(sinr)
         inter = [str(i) for i in pos[1]]
         inter = ','.join(inter)
         inter = '\'' + inter
         dataframe = pd.DataFrame({'throughput': [thr], 'interference': [inter], 'RSSI': [rssi], 'SINR': [sinr]})
         path="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\output_2\\"
         mkdir(path)
         dataframe.to_csv(path+input_file_name + ".csv", index=False, sep=',')
         if count2==21:
             s+=1
             count2=0
         count1=0
         pos=[]





