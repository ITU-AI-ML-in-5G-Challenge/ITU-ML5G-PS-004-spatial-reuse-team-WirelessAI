import pandas as pd
import os
import glob

file_path="C:\\Users\\18545\\Desktop\\simulator_input_files"
os.chdir(file_path)

for counter, current_file in enumerate(glob.glob("*.csv")):
    d = pd.read_csv(current_file, sep = ';', header = None)
    d.to_csv('C:\\Users\\18545\\Desktop\\Context\\'+current_file,header=None,index=False)