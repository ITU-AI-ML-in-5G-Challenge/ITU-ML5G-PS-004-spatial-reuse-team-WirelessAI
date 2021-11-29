import pandas as pd
import os
import glob

file_path="C:\\Users\\18545\\Desktop\\Context\\input"
os.chdir(file_path)
count=0
context_id=0


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

for counter, current_file in enumerate(glob.glob("*.csv")):
    d = pd.read_csv(current_file)
    count+=1
    node_code=d.loc[:,'node_code']
    node_type=d.loc[:,'node_type']
    wlan_code=d.loc[:,'wlan_code']
    x=d.loc[:,'x(m)']
    y = d.loc[:, 'y(m)']
    dataframe = pd.DataFrame({'node_code': node_code, 'node_type': node_type, 'wlan_code': wlan_code, 'x(m)': x,'y(m)':y})

    if count==21 or context_id==0:
        path="C:\\Users\\18545\\Desktop\\Context\\input_processed\\Context"+str(context_id)+"\\"
        context_id += 1
        mkdir(path)
        count=0
    dataframe.to_csv(path + current_file, index=False,sep=',')