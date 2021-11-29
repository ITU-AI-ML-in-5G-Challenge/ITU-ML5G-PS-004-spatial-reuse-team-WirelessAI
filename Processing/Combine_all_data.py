import pandas as pd
import os
import glob


path="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\new_train_data"
os.chdir(path)
file=[]
for counter, current_file in enumerate(glob.glob("*.csv")):
    print(current_file)
    input = pd.read_csv(current_file)
    file.append(input)
train = pd.concat(file, axis=0)
train_path="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\new_train_data\\"
train.to_csv(train_path+'All_combine_data' + ".csv", index=0, sep=',')
