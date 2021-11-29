# ITU-ML5G-PS-004: Federated Learning for Spatial Reuse in a Multi-BSS (Basic Service Set) Scenario
This repo contains our submission for Problem 004 of the AI for Good Machine Learning in 5G Challenge (i.e., [ITUChallenge 2021](https://challenge.aiforgood.itu.int/match/matchitem/37)). 

## Authors

**Team Name:** WirelessAI

**Team Members:**
- Mr. Hao Chen, Xiamen University, China, Email: chenhao24@stu.xmu.edu.cn
- Ms. Xiaoying Ye, Xiamen University, China, Email: yexiaoying0714@stu.xmu.edu.cn
- Dr. Lizhao You, Xiamen University, China, Email: lizhaoyou@xmu.edu.cn
- Dr. Yulin Shao, Imperial College London, Email: y.shao@imperial.ac.uk

## Problem

  The problem statement information can be found at https://www.upf.edu/web/wnrg/2021-edition. This problem statement is framed into the Networks-track because it gives rise to develop ML models based on a training dataset that will be provided to participants.

## Dataset
- A dataset generated with the Komondor simulator [6] is provided to train FL models. You can access the dataset at https://zenodo.org/record/5352060#.YYKrxI4zZPZ.
- Before training, we pre-process the dataset through codes in Processing folder. We run files named *Extract_input.py* and *Extract_out.py* to get input and output data we need, and then we run the file named *Combine_output_input.py* to combine the output and input data in the same file. 
- The pre-processed training data (we only use training data 3 for training) and test data files are *train_data_3.rar* and *test_data.rar*. The pre-processed dataset can be downloaded from Google Cloud. The URL is https://drive.google.com/file/d/1FkjULnWZVzTEUoI25sTcUx9UrpejQ56p/view?usp=sharing.

## Model
- **FL+CNN+FCNN**: We used federated learning (FL), a combination of convolutional neural network (CNN) to predict interference, SINR, RSSI, and as input to the fully connected neural network (FCNN) to predict throughput, in order to find a suitable PD value to solve this problem. 
- After we process the raw data through Processing, we save the model in files, and then use the model to predicate the performance of test data.
- The model we have trained can be downloaded from Google Cloud. The URL is https://drive.google.com/file/d/1FkjULnWZVzTEUoI25sTcUx9UrpejQ56p/view?usp=sharing.


## Requirments

Install all the packages:
- pytorch
- python3

## Running the Codes

The experiment trains the model following the conventional federated training way, which involves with training a global model using many local models.

- To run the federated training experiment base on training dataset with FL on cnn and fc using GPU:

```python FL/work3.py --global_ep=20 --gpu=0```

- To run the federated testting experiment based on test dataset with FL on CNN and FC(NIID)

```python FL/test.py```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options

The default values for various paramters parsed to the experiment are given in the following. Details are given some of those parameters:

```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.

```--global_ep:```   Number of rounds of global training.

```--learning_rate:```       Learning rate set to 1e-5 by default.

### Federated Options

```--train_len:```       Number of training contexts.

```--test_len:```       Number of testing contexts.

```--local_ep:```   Number of rounds of local training.

## Training Parameters

The experiment involves with training a global model in the federated setting.

Federated parameters (default values):

* ```Fraction of users (C)```: 0.02
* ```Local Batch size  (B)```: 16 
* ```Local_ep      ```: 400
* ```global_ep      ```: 10
* ```Optimizer            ```: Adam 
* ```Learning Rate        ```: 1e-5 <br />

## Results 

You can find the final results in *output_11ax_sr_simulations_test.txt*.

