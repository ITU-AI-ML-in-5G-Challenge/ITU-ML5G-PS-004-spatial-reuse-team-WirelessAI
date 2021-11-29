import pandas as pd
import os
import glob
import csv
import re

file_path="C:\\Users\\18545\\Desktop\\ITU challenge\\Context\\input"
os.chdir(file_path)
xmax=0
xmin=0
for counter, current_file in enumerate(glob.glob("*.csv")):
    d = pd.read_csv(current_file)
    check_data = d.loc[:, 'x(m)']
    check_data = list(map(float, check_data))
    for i in range(len(check_data)):
        if check_data[i]>xmax:
            xmax=check_data[i]
        if check_data[i]<xmin:
            xmin=check_data[i]
print(xmin,xmax)




