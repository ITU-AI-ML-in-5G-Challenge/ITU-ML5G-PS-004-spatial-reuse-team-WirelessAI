import pandas as pd
import os
import glob

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



for context_id in range(3000,4000):
    path_input="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\input_2\\Context"+str(context_id)+"\\"
    path_output="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\output_2\\Context"+str(context_id)+"\\"
    os.chdir(path_input)
    obss_pd=-62
    for counter, current_file in enumerate(glob.glob("*.csv")):
        input = pd.read_csv(current_file)
        output_file_name='sim_'+current_file.strip('.csv')+'_output'+'.csv'
        output=pd.read_csv(path_output+output_file_name)
        file = [input,output]
        train = pd.concat(file, axis=1)
        train['obss/pd']=obss_pd
        train_path="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\train_data_2\\Context"+str(context_id)+"\\"
        mkdir(train_path)
        train.to_csv(train_path+"train_data_s"+str(context_id)+'_'+str(obss_pd) + ".csv", index=0, sep=',')
        obss_pd -= 1
        if obss_pd==-83 :
            obss_pd=-62

