import pandas as pd
import os
import glob
import numpy as np
import csv
from tqdm import tqdm
import re
import scipy.io as sio
pwd="C:\ITUchallenge3"
class train_dataset():
    def __init__(self):
       self.input_data1 = []
       self.label_data1 = []
       self.label_data2=[]
       self.count=0
    def process_data(self,j):
        self.input_data1 = []
        self.label_data1 = []
        self.label_data2 = []
        self.count = 0
        for i in tqdm(range(j,j+1)):
            file_path = "C:\\ITUchallenge3\\train_data_3\\Context"+str(i)
            os.chdir(file_path)
            for counter, current_file in enumerate(glob.glob("*.csv")):
                topo_image = np.zeros((1, 100, 100))
                with open(current_file, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]
                    for i in range(1, len(rows)):
                        if rows[i][0] == 'AP_A':
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            topo_image[0, x, y] =abs(int(rows[i][9]))
                            interference=rows[i][6]
                            interference=re.findall(r"\d+\.?\d*",interference)#统计inter里的数值
                            interference=list(map(float,interference))
                            interference=np.array([-x for x in interference])
                            interference = np.pad(interference, (0, 5 - interference.shape[0]), 'constant', constant_values=(0, 0))#补零)
                            rssi=rows[i][7]
                            sinr=rows[i][8]
                            rssi = re.findall(r"\d+\.?\d*", rssi)
                            rssi = list(map(float, rssi))
                            rssi=np.array(rssi)
                            rssi = np.pad(rssi, (0, 6 - rssi.shape[0]), 'constant',
                                                  constant_values=(0, 0))

                            sinr=sinr.split(",")
                            sinr = list(map(float, sinr))
                            sinr = np.array(sinr)
                            sinr = np.pad(sinr, (0, 6 - sinr.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            if rows[i][8]!='':
                                label1=list(interference)+list(rssi)+list(sinr)
                                #label1 = [float(rows[i][7])] + [float(rows[i][8])]
                            else:
                                label1=list(interference)+list(rssi)+[0,0,0,0,0,0]
                                #label1 = [float(rows[i][7])] + [0]
                            label1=np.array(label1)
                            #print(label1)
                            thr=rows[i][5]
                            thr=thr.split(",")
                            label2=np.array(list(map(float,thr)))
                            label2 = np.pad(label2, (0, 6 - label2.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            self.label_data1.append(label1)
                            self.label_data2.append(label2)
                        elif rows[i][2]=='A':
                                x = float(rows[i][3])
                                y = float(rows[i][4])
                                x = round(x+10)
                                y = round(y+10)
                                topo_image[0, x, y] = 3
                        else:
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            if rows[i][1] == '0':
                                topo_image[0, x, y] = 1
                            else:
                                topo_image[0, x, y] = 2
                    self.input_data1.append(topo_image)

        self.input_data1 = np.array(self.input_data1)
        self.label_data1=np.array(self.label_data1)
        self.label_data2 = np.array(self.label_data2)
        self.count=counter
        return self.input_data1,self.label_data1,self.label_data2,self.count

#test=train_dataset()
#test.process_data()
class test_dataset():
    def __init__(self):
       self.input_data1 = []
       self.label_data1 = []
       self.label_data2=[]
       self.count=0
    def process_data(self,j):
        self.input_data1 = []
        self.label_data1 = []
        self.label_data2 = []
        self.count=0

        for i in tqdm(range(j,j+1)):
            file_path = "C:\\ITUchallenge3\\train_data_3\\Context"+str(i)
            os.chdir(file_path)
            for counter, current_file in enumerate(glob.glob("*.csv")):
                topo_image = np.zeros((1, 100, 100))
                with open(current_file, 'r',encoding='UTF-8') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]
                    for i in range(1, len(rows)):
                        if rows[i][0] == 'AP_A':
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            topo_image[0, x, y] =abs(int(rows[i][9]))
                            interference=rows[i][6]
                            interference=re.findall(r"\d+\.?\d*",interference)#统计inter里的数值
                            interference=list(map(float,interference))
                            interference=np.array([-x for x in interference])
                            interference = np.pad(interference, (0, 5 - interference.shape[0]), 'constant', constant_values=(0, 0))#补零)
                            rssi=rows[i][7]
                            sinr=rows[i][8]
                            rssi = re.findall(r"\d+\.?\d*", rssi)
                            rssi = list(map(float, rssi))
                            rssi=np.array(rssi)
                            rssi = np.pad(rssi, (0, 6 - rssi.shape[0]), 'constant',
                                                  constant_values=(0, 0))

                            sinr=sinr.split(",")
                            sinr = list(map(float, sinr))
                            sinr = np.array(sinr)
                            sinr = np.pad(sinr, (0, 6 - sinr.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            if rows[i][8]!='':
                                label1=list(interference)+list(rssi)+list(sinr)
                                #label1 = [float(rows[i][7])] + [float(rows[i][8])]
                            else:
                                label1=list(interference)+list(rssi)+[0,0,0,0,0,0]
                                #label1 = [float(rows[i][7])] + [0]
                            label1=np.array(label1)
                            #print(label1)
                            thr=rows[i][5]
                            thr=thr.split(",")
                            label2=np.array(list(map(float,thr)))
                            label2 = np.pad(label2, (0, 6 - label2.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            self.label_data1.append(label1)
                            self.label_data2.append(label2)
                        elif rows[i][2]=='A':
                                x = float(rows[i][3])
                                y = float(rows[i][4])
                                x = round(x+10)
                                y = round(y+10)
                                topo_image[0, x, y] = 3
                        else:
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            if rows[i][1] == '0':
                                topo_image[0, x, y] = 1
                            else:
                                topo_image[0, x, y] = 2
                    self.input_data1.append(topo_image)


        self.count=counter
        self.input_data1 = np.array(self.input_data1)
        self.label_data1=np.array(self.label_data1)
        self.label_data2 = np.array(self.label_data2)
        return self.input_data1,self.label_data1,self.label_data2,self.count

class test_dataset1():
    def __init__(self):
       self.input_data1 = []
       self.count=0
       self.t=0

    def process_data(self,j):
        self.input_data1 = []
        self.count = 0
        self.t=0

        for i in tqdm(range(j,j+1)):
            file_path = "C:\\ITUchallenge3\\test_data\\Context"+str(i)
            os.chdir(file_path)
            k=0
            for counter, current_file in enumerate(glob.glob("*.csv")):
                topo_image = np.zeros((1, 100, 100))
                with open(current_file, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]
                    for i in range(1, len(rows)):
                        if rows[i][0] == 'AP_A':
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            topo_image[0, x, y] =abs(int(rows[i][9]))
                            interference=rows[i][6]
                            interference=re.findall(r"\d+\.?\d*",interference)#统计inter里的数值
                            interference=list(map(float,interference))
                            interference=np.array([-x for x in interference])
                            interference = np.pad(interference, (0, 5 - interference.shape[0]), 'constant', constant_values=(0, 0))#补零)
                        elif rows[i][2]=='A':
                                x = float(rows[i][3])
                                y = float(rows[i][4])
                                x = round(x+10)
                                y = round(y+10)
                                k+=1
                                topo_image[0, x, y] = 3
                        else:
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            if rows[i][1] == '0':
                                topo_image[0, x, y] = 1
                            else:
                                topo_image[0, x, y] = 2
                    self.input_data1.append(topo_image)
                    counter+=1
        self.count = counter
        self.t=k
        self.input_data1 = np.array(self.input_data1)

        return self.input_data1,self.count,self.t

class test_dataset2():
    def __init__(self):
       self.input_data1 = []
       self.count=0
       self.t=0

    def process_data(self,j):
        self.input_data1 = []
        self.count = 0
        self.t=0

        for i in tqdm(range(j,j+1)):
            file_path = "C:\\ITUchallenge3\\train_data_3\\Context"+str(i)
            os.chdir(file_path)
            k=0
            for counter, current_file in enumerate(glob.glob("*.csv")):
                topo_image = np.zeros((1, 100, 100))
                with open(current_file, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]
                    for i in range(1, len(rows)):
                        if rows[i][0] == 'AP_A':
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            topo_image[0, x, y] =abs(int(rows[i][9]))
                            interference=rows[i][6]
                            interference=re.findall(r"\d+\.?\d*",interference)#统计inter里的数值
                            interference=list(map(float,interference))
                            interference=np.array([-x for x in interference])
                            interference = np.pad(interference, (0, 5 - interference.shape[0]), 'constant', constant_values=(0, 0))#补零)
                        elif rows[i][2]=='A':
                                x = float(rows[i][3])
                                y = float(rows[i][4])
                                x = round(x+10)
                                y = round(y+10)
                                k+=1
                                topo_image[0, x, y] = 3
                        else:
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            if rows[i][1] == '0':
                                topo_image[0, x, y] = 1
                            else:
                                topo_image[0, x, y] = 2
                    self.input_data1.append(topo_image)
                    counter+=1
        self.count = counter
        self.t=k
        self.input_data1 = np.array(self.input_data1)

        return self.input_data1,self.count,self.t


class train_dataset0():
    def __init__(self):
       self.input_data1 = []
       self.label_data1 = []
       self.label_data2=[]
       self.count=0
       self.t=[]
    def process_data(self,j):
        self.input_data1 = []
        self.label_data1 = []
        self.label_data2 = []
        self.count=0
        self.t=[]

        for i in tqdm(range(j,j+1)):

            file_path = "C:\\ITUchallenge3\\train_data_3\\Context"+str(i)
            os.chdir(file_path)
            for counter, current_file in enumerate(glob.glob("*.csv")):
                topo_image = np.zeros((1, 101, 101))
                k = 0
                with open(current_file, 'r',encoding='UTF-8') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = [row for row in reader]
                    for i in range(1, len(rows)):
                        if rows[i][0] == 'AP_A':
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            topo_image[0, x, y] =abs(int(rows[i][9]))
                            interference=rows[i][6]
                            interference=re.findall(r"\d+\.?\d*",interference)#统计inter里的数值
                            interference=list(map(float,interference))
                            interference=np.array([-x for x in interference])
                            interference = np.pad(interference, (0, 5 - interference.shape[0]), 'constant', constant_values=(0, 0))#补零)
                            rssi=rows[i][7]
                            sinr=rows[i][8]
                            rssi = re.findall(r"\d+\.?\d*", rssi)
                            rssi = list(map(float, rssi))
                            rssi=np.array(rssi)
                            rssi = np.pad(rssi, (0, 6 - rssi.shape[0]), 'constant',
                                                  constant_values=(0, 0))

                            sinr=sinr.split(",")
                            sinr = list(map(float, sinr))
                            sinr = np.array(sinr)
                            sinr = np.pad(sinr, (0, 6 - sinr.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            if rows[i][8]!='':
                                label1=list(interference)+list(rssi)+list(sinr)
                                #label1 = [float(rows[i][7])] + [float(rows[i][8])]
                            else:
                                label1=list(interference)+list(rssi)+[0,0,0,0,0,0]
                                #label1 = [float(rows[i][7])] + [0]
                            label1=np.array(label1)
                            #print(label1)
                            thr=rows[i][5]
                            thr=thr.split(",")
                            label2=np.array(list(map(float,thr)))
                            label2 = np.pad(label2, (0, 6 - label2.shape[0]), 'constant',
                                          constant_values=(0, 0))
                            self.label_data1.append(label1)
                            self.label_data2.append(label2)
                        elif rows[i][2]=='A':
                                x = float(rows[i][3])
                                y = float(rows[i][4])
                                x = round(x+10)
                                y = round(y+10)
                                topo_image[0, x, y] = 3
                                k+=1
                        else:
                            x = float(rows[i][3])
                            y = float(rows[i][4])
                            x = round(x+10)
                            y = round(y+10)
                            if rows[i][1] == '0':
                                topo_image[0, x, y] = 1
                            else:
                                topo_image[0, x, y] = 2
                    self.input_data1.append(topo_image)
                    self.t.append(k)



        self.count=counter+1
        self.t=list(self.t)
        self.input_data1 = np.array(self.input_data1)
        self.label_data1=np.array(self.label_data1)
        self.label_data2 = np.array(self.label_data2)
        return self.input_data1,self.label_data1,self.label_data2,self.count,self.t