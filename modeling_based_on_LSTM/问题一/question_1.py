import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
import math
import xgboost
import time
from tqdm import tqdm
import keras.models
import keras.layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import warnings
warnings.simplefilter('ignore')

data_train_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\data1_trainset.txt"
data_test_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\data1_testset.txt"
data_RUL_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\data1_psi.txt"

data_train = pd.read_csv(data_train_file,sep=",",header=None)
data_test = pd.read_csv(data_test_file,sep=",",header=None)
data_RUL = pd.read_csv(data_RUL_file,sep=",",header=None)

train_copy = data_train
test_copy = data_test
data_train.drop(0,inplace=True)
data_test.drop(0,inplace=True)
data_RUL.drop(0,inplace=True)
#数据预处理
columns_train = ['unit_ID','cycles','bitDepth','WOB','holeDepth','f','psi','w','torque','s1','s2' ]
data_train.columns = columns_train
data_train.describe()
data_train['unit_ID']=pd.to_numeric(data_train['unit_ID'],errors='coerce')
data_train['cycles']=pd.to_numeric(data_train['cycles'],errors='coerce')
data_train['bitDepth']=pd.to_numeric(data_train['bitDepth'],errors='coerce')
data_train['WOB']=pd.to_numeric(data_train['WOB'],errors='coerce')
data_train['holeDepth']=pd.to_numeric(data_train['holeDepth'],errors='coerce')
data_train['f']=pd.to_numeric(data_train['f'],errors='coerce')
data_train['psi']=pd.to_numeric(data_train['psi'],errors='coerce')
data_train['w']=pd.to_numeric(data_train['w'],errors='coerce')
data_train['torque']=pd.to_numeric(data_train['torque'],errors='coerce')
data_train['s1']=pd.to_numeric(data_train['s1'],errors='coerce')
data_train['s2']=pd.to_numeric(data_train['s2'],errors='coerce')
# Define a function to calculate the SPP（psi)
def add_psi(g):
    g['SPP'] = (g['psi']).mean()
    return g
train = data_train.groupby('unit_ID').apply(add_psi)

def process_input_data_with_targets(input_data, target_data = None, window_length = 1, shift = 1):
    num_batches = np.int(np.floor((len(input_data) - window_length)/shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches, window_length,
                                                                                                  num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats = num_batches)
        for batch in range(num_batches):
            output_data[batch,:,:] = input_data[(0+shift*batch):(0+shift*batch+window_length),:]
            output_targets[batch] = target_data[(shift*batch + (window_length-1))]
        return output_data, output_targets


def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=5):#五步取平均
    max_num_test_batches = np.int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows
test_data_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\data1_testset.txt"
true_rul_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\data1_psi.txt"
train_target_file="C:\\Users\\99080\\PycharmProjects\\pythonProject\\train_psi.txt"
test_data = pd.read_csv(test_data_file, sep = ",", header = None,names=columns_train )
true_psi = pd.read_csv(true_rul_file, sep = ',', header = None,names=["psi"])
train_target=pd.read_csv(train_target_file,sep=',',names=["target"])

#测试数据处理
test_data.drop(0,inplace=True)
true_psi.drop(0,inplace=True)
train_target.drop(0,inplace=True)
true_psi['psi']=pd.to_numeric(true_psi['psi'],errors='coerce')
train_target['target']=pd.to_numeric(train_target['target'],errors='coerce')
test_data['unit_ID']=pd.to_numeric(test_data['unit_ID'],errors='coerce')
test_data['cycles']=pd.to_numeric(test_data['cycles'],errors='coerce')
test_data['bitDepth']=pd.to_numeric(test_data['bitDepth'],errors='coerce')
test_data['WOB']=pd.to_numeric(test_data['WOB'],errors='coerce')
test_data['holeDepth']=pd.to_numeric(test_data['holeDepth'],errors='coerce')
test_data['f']=pd.to_numeric(test_data['f'],errors='coerce')
test_data['psi']=pd.to_numeric(test_data['psi'],errors='coerce')
test_data['w']=pd.to_numeric(test_data['w'],errors='coerce')
test_data['torque']=pd.to_numeric(test_data['torque'],errors='coerce')
test_data['s1']=pd.to_numeric(test_data['s1'],errors='coerce')
test_data['s2']=pd.to_numeric(test_data['s2'],errors='coerce')
window_length = 20#30->20
shift = 1#1
# early_rul = 125
processed_train_data = []
processed_train_targets = []
num_test_windows = 5#5
processed_test_data = []
num_test_windows_list = []

columns_to_be_dropped =['unit_ID','holeDepth','psi','s1', 's2']#这个是为了删data_train里的

train_data_first_column = data_train ["unit_ID"]
test_data_first_column = test_data["unit_ID"]

#test_data是文本没有影响
scaler = StandardScaler()
train_data = scaler.fit_transform(data_train.drop(columns = columns_to_be_dropped))
test_data = scaler.transform(test_data.drop(columns = columns_to_be_dropped))

train_data = pd.DataFrame(data = np.c_[train_data_first_column, train_data])
test_data = pd.DataFrame(data = np.c_[test_data_first_column, test_data])

num_train_machines = len(train_data[0].unique())#因为是自己删了一行，所以要减一
num_test_machines = len(test_data[0].unique())
def process_targets(data_length,idx):
    temp=train_target.loc[idx,'target']
    return [temp]*data_length
for i in np.arange(1, num_train_machines + 1):
    temp_train_data = train_data[train_data[0] == i].drop(columns=[0]).values
    # temp_train_data = train_data[train_data[0] == 1].drop(columns=[0]).values
    # Determine whether it is possible to extract training data with the specified window length.
    if (len(temp_train_data) < window_length):
        print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    temp_train_targets = process_targets(data_length=temp_train_data.shape[0], idx=i)
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets,
                                                                                window_length=window_length,
                                                                                shift=shift)

    processed_train_data.append(data_for_a_machine)
    processed_train_targets.append(targets_for_a_machine)

processed_train_data = np.concatenate(processed_train_data)
processed_train_targets = np.concatenate(processed_train_targets)
for i in np.arange(1, num_test_machines + 1):
    temp_test_data = test_data[test_data[0] == i].drop(columns=[0]).values

    # Determine whether it is possible to extract test data with the specified window length.
    if (len(temp_test_data) < window_length):
        print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    # Prepare test data
    test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift=shift,
                                                             num_test_windows=num_test_windows)

    processed_test_data.append(test_data_for_an_engine)
    num_test_windows_list.append(num_windows)

processed_test_data = np.concatenate(processed_test_data)
true_psi = true_psi['psi'].values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]
print("Processed trianing data shape: ", processed_train_data.shape)
print("Processed training ruls shape: ", processed_train_targets.shape)
print("Processed test data shape: ", processed_test_data.shape)
print("True RUL shape: ", true_psi.shape)
processed_train_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_train_data,
                                                                                                            processed_train_targets,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 83)
print("Processed train data shape: ", processed_train_data.shape)
print("Processed validation data shape: ", processed_val_data.shape)
print("Processed train targets shape: ", processed_train_targets.shape)
print("Processed validation targets shape: ", processed_val_targets.shape)

def create_compiled_model():
    model = Sequential([
        layers.LSTM(128, input_shape = (window_length, 6), return_sequences=True, activation = "tanh"),
        layers.LSTM(64, activation = "tanh", return_sequences = True),
        layers.LSTM(32, activation = "tanh"),
        layers.Dense(96, activation = "relu"),
        layers.Dense(128, activation = "relu"),
        layers.Dense(1)
    ])
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

def scheduler(epoch):
    if epoch < 20:
        return 0.01
    elif (epoch<70&epoch>=20):
        return 0.001
    else:
        return 0.0001
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
model = create_compiled_model()
history = model.fit(processed_train_data, processed_train_targets, epochs = 120,#确认错误后改epoch个数
                    validation_data = (processed_val_data, processed_val_targets),
                    callbacks = callback,
                    batch_size = 64, verbose = 2)
'''
tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=True)
file_writer=tf.summary.create_file_writer("./logs")
with file_writer.as_default():
    tf.summary.image("Model Architecture",
                     tf.keras.preprocessing.image.load_img('model.png'),
                     step=0)'''
plt.plot(history.history['loss'],label="Train Loss",color="salmon")
plt.plot(history.history['val_loss'],label='Val Loss',color='turquoise')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.show()

rul_pred = model.predict(processed_test_data).reshape(-1)
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
RMSE = np.sqrt(mean_squared_error(true_psi, mean_pred_for_each_engine))
print("RMSE: ", RMSE)
tf.keras.models.save_model(model, "data1_LSTM_piecewise_RMSE_"+ str(np.round(RMSE, 4)) + ".h5")

indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]

RMSE_new = np.sqrt(mean_squared_error(true_psi, preds_for_last_example))
print("RMSE (Taking only last examples): ", RMSE_new)

plt.figure()
plt.plot(true_psi, label = "True PSI", color = "cornflowerblue")
plt.plot(preds_for_last_example, label = "Pred PSI", color = "coral")
plt.legend()
plt.xlabel("timeStep")
plt.ylabel("SPP(psi)")
plt.show()
