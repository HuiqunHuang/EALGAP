# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptBikeNYC.py
"""

from __future__ import print_function
import os
import _pickle as pickle
import random
import time

import numpy as np
import math
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import get_default_graph
from datetime import datetime as dt
from DataProcessing.PrepareDataForModel import loadDataForModel_NYC
from FileOperation.LoadH5 import return_threshold
from Model.NYC_EALGAP_Model import RegionalDataPredictionModel
from config import Config
import metrics as metrics
import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1337
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.python.keras.backend import set_session
sess = Session()
graph = get_default_graph()
set_session(sess)

DATAPATH = Config().DATAPATH
nb_epoch_cont = 500
batch_size = 96
T = 24
lr = 0.002
len_closeness = 5
len_distribution = 1 * T
pdf_ratio = 1000
len_day = 3
len_trend = 2
days_test = 10
days_val = 5
len_test = T * days_test
len_val = T * days_val
categorynum = 20
clusternum = 20 # region num
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
embedding_length = 5
d = 1

def build_model():
    model = RegionalDataPredictionModel(node_num=clusternum, len_closeness=len_closeness, len_day=len_day, T=T,
                                        embedding_length=embedding_length, d=d)
    return model

@tf.function
def train_step_function(model, x_batch_train, y_batch_train, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        mse = metrics.mean_squared_error(y_batch_train[0], logits[0])
    grads = tape.gradient(mse, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return mse

@tf.function
def val_step_function(x_batch_val, y_batch_val, model):
    val_logits = model(x_batch_val, training=False)
    val_mse = metrics.mean_squared_error(y_batch_val[0], val_logits[0])
    return val_mse

def main(data_file_name, city, project_path, extreme_high_valid_percent, extreme_low_valid_percent, datatype="nyc bike 2020"):
    print("loading data...")
    extreme_high, extreme_low = return_threshold(project_path + "Data\\NYC\\" + data_file_name, extreme_high_valid_percent, extreme_low_valid_percent)
    print("Extreme high: " + str(extreme_high) + ", extreme low: " + str(extreme_low))

    X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, \
    near_category_X_train, near_category_X_val, near_category_X_test, \
    near_category_y_train, near_category_y_val, near_category_y_test, \
    extreme_data_X_train, extreme_data_X_val, extreme_data_X_test, \
    extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test, \
    label_Y_train, label_Y_val, label_Y_test, \
    target_label_Y_train, target_label_Y_val, target_label_Y_test, \
    exponential_data_X_train, exponential_data_X_val, exponential_data_X_test, \
    same_hour_mean_X_train, same_hour_mean_X_val, same_hour_mean_X_test, \
    same_hour_var_X_train, same_hour_var_X_val, same_hour_var_X_test, \
    same_hour_mean_Y_train, same_hour_mean_Y_val, same_hour_mean_Y_test, \
    same_hour_var_Y_train, same_hour_var_Y_val, same_hour_var_Y_test, \
    window_same_hour_mean_X_train, window_same_hour_mean_X_val, window_same_hour_mean_X_test, \
    window_same_hour_mean_Y_train, window_same_hour_mean_Y_val, window_same_hour_mean_Y_test, \
    window_same_hour_var_X_train, window_same_hour_var_X_val, window_same_hour_var_X_test, \
    window_same_hour_var_Y_train, window_same_hour_var_Y_val, window_same_hour_var_Y_test, \
    window_data_X_train, window_data_X_val, window_data_X_test, \
    window_data_Y_train, window_data_Y_val, window_data_Y_test, \
    window_exponential_data_X_train, window_exponential_data_X_val, window_exponential_data_X_test, \
    spearson_data_X_train, spearson_data_X_val, spearson_data_X_test, \
    window_spearson_data_X_train, window_spearson_data_X_val, window_spearson_data_X_test = \
        loadDataForModel_NYC(datatype, data_file_name, T=T, len_closeness=len_closeness, len_trend=len_trend,
                             len_distribution=len_distribution, pdf_ratio=pdf_ratio, len_day=len_day, len_test=len_test,
                             len_val=len_val, extreme_high=extreme_high)

    near_category_X_train = near_category_X_train.astype('float32')
    near_category_y_train = near_category_y_train.astype('float32')
    exponential_data_X_train = exponential_data_X_train.astype('float32')
    same_hour_mean_X_train = same_hour_mean_X_train.astype('float32')
    same_hour_var_X_train = same_hour_var_X_train.astype('float32')
    window_same_hour_mean_X_train = window_same_hour_mean_X_train.astype('float32')
    window_same_hour_mean_Y_train = window_same_hour_mean_Y_train.astype('float32')
    window_same_hour_var_X_train = window_same_hour_var_X_train.astype('float32')
    window_same_hour_var_Y_train = window_same_hour_var_Y_train.astype('float32')
    window_data_X_train = window_data_X_train.astype('float32')
    window_data_Y_train = window_data_Y_train.astype('float32')
    window_exponential_data_X_train = window_exponential_data_X_train.astype('float32')

    train_dataset_X = tf.data.Dataset.from_tensor_slices((near_category_X_train, exponential_data_X_train,
                                                          same_hour_mean_X_train, same_hour_var_X_train,
                                                          window_same_hour_mean_X_train,
                                                          window_same_hour_var_X_train,
                                                          window_data_X_train,
                                                          window_exponential_data_X_train))
    train_dataset_Y = tf.data.Dataset.from_tensor_slices((near_category_y_train, window_data_Y_train))
    train_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_Y)).shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    near_category_X_val = near_category_X_val.astype('float32')
    near_category_y_val = near_category_y_val.astype('float32')
    exponential_data_X_val = exponential_data_X_val.astype('float32')
    same_hour_mean_X_val = same_hour_mean_X_val.astype('float32')
    same_hour_var_X_val = same_hour_var_X_val.astype('float32')
    window_same_hour_mean_X_val = window_same_hour_mean_X_val.astype('float32')
    window_same_hour_mean_Y_val = window_same_hour_mean_Y_val.astype('float32')
    window_same_hour_var_X_val = window_same_hour_var_X_val.astype('float32')
    window_same_hour_var_Y_val = window_same_hour_var_Y_val.astype('float32')
    window_data_X_val = window_data_X_val.astype('float32')
    window_data_Y_val = window_data_Y_val.astype('float32')
    window_exponential_data_X_val = window_exponential_data_X_val.astype('float32')

    val_dataset_X = tf.data.Dataset.from_tensor_slices((near_category_X_val, exponential_data_X_val,
                                                        same_hour_mean_X_val, same_hour_var_X_val,
                                                        window_same_hour_mean_X_val,
                                                        window_same_hour_var_X_val,
                                                        window_data_X_val,
                                                        window_exponential_data_X_val))
    val_dataset_Y = tf.data.Dataset.from_tensor_slices((near_category_y_val, window_data_Y_val))
    val_dataset = tf.data.Dataset.zip((val_dataset_X, val_dataset_Y)).shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).repeat(100)


    model = build_model()
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    train_mse_all = []
    val_mse_all = []
    train_window_mse_all = []
    val_window_mse_all = []

    start = dt.now()

    best_mse = 1000000000000
    hyperparams_name = 'c{}.len_day{}.lr{}.batchsize{}'.format(len_closeness, len_day, lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    val_best_mse = 10000000000000

    for epoch in range(nb_epoch_cont):
        # print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        train_total_mse = 0
        val_total_mse = 0

        train_step = 0
        val_step = 0
        for x_batch_train, y_batch_train in train_dataset:
            mse = train_step_function(model, x_batch_train, y_batch_train, optimizer)
            train_total_mse += mse
            train_step += 1
            if mse < best_mse:
                best_mse = mse

        for x_batch_val, y_batch_val in val_dataset:
            val_mse = val_step_function(x_batch_val, y_batch_val, model)
            val_total_mse += val_mse
            val_step += 1

        if val_total_mse / val_step < val_best_mse:
            val_best_mse = val_total_mse / val_step
            model.save_weights(os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name)), overwrite=True)
            model.load_weights(fname_param)

        train_mse_all.append(train_total_mse / train_step)
        val_mse_all.append(val_total_mse / val_step)
        print('| epoch {:3d} | time: {:5.2f}s | train_mse {:5.4f} | val_mse {:5.4f} '.format(
            epoch, (time.time() - start_time), train_total_mse / train_step, val_total_mse / val_step))

    total_mae = 0
    total_mse = 0
    total_msle = 0
    total_mape = 0
    length_total = 0
    total_ground_truth = 0

    inverse_data = []
    before_inverse = []
    original_data = []
    # print(np.array(X_data_test).shape)
    y_predict = model.predict(X_data_test)
    # window_pre = y_predict[1]
    Q = y_predict[1]
    K = y_predict[2]
    V = y_predict[3]
    QK = y_predict[4]
    data_input = y_predict[5]
    attention_score = y_predict[6]
    extreme_probability = y_predict[7]
    pre_extreme_degree = y_predict[8]
    same_hour_mean = y_predict[9]
    same_hour_var = y_predict[10]
    total_dis = 0
    mean_gro = np.mean(np.array(Y_data_test[0]))

    y_predict = y_predict[0]
    print(np.array(target_label_Y_test).shape)
    print(np.array(y_predict).shape)
    # print(np.array(Y_data_test).shape)
    print(np.array(target_label_Y_test).shape)
    print("Q: " + str(np.array(Q).shape))
    print("K: " + str(np.array(K).shape))
    print("V: " + str(np.array(V).shape))
    print("QK: " + str(np.array(QK).shape))
    print("data_input: " + str(np.array(data_input).shape))
    print("attention_score: " + str(np.array(attention_score).shape))
    print("extreme_probability: " + str(np.array(extreme_probability).shape))
    print("pre_extreme_degree: " + str(np.array(pre_extreme_degree).shape))

    Q_new, K_new, V_new, data_input_new, attention_score_new, extreme_probability_new, \
    nor_data_new, same_hour_mean_new, same_hour_var_new = [], [], [], [], [], [], [], [], []
    pre_extreme_degree_new, same_hour_mean_Y_new, same_hour_var_Y_new = [], [], []
    gamma_new, beta_new = [], []
    first_row = []
    for i in range(0, categorynum):
        Q_new.append([])
        K_new.append([])
        V_new.append([])
        data_input_new.append([])
        attention_score_new.append([])
        extreme_probability_new.append([])
        first_row.append("Region " + str(i + 1))
        nor_data_new.append([])
        same_hour_mean_new.append([])
        same_hour_var_new.append([])
        gamma_new.append([])
        beta_new.append([])
        pre_extreme_degree_new.append([])
        same_hour_mean_Y_new.append([])
        same_hour_var_Y_new.append([])
    for i in range(0, len(Q)):
        for j in range(0, len(Q[i])):
            for k in range(0, len(Q[i][j])):
                Q_new[j].append(Q[i][j][k][0])
                K_new[j].append(K[i][j][k][0])
                V_new[j].append(V[i][j][k][0])
                data_input_new[j].append(data_input[i][j][k][0])
                extreme_probability_new[j].append(extreme_probability[i][j][k])
                for m in range(0, len(attention_score[i][j][k])):
                    attention_score_new[j].append(attention_score[i][j][k][m])
            pre_extreme_degree_new[j].append(pre_extreme_degree[i][j])
            same_hour_mean_Y_new[j].append(same_hour_mean_Y_test[i][j])
            same_hour_var_Y_new[j].append(same_hour_var_Y_test[i][j])

    for i in range(0, len(y_predict)):
        for j in range(0, len(target_label_Y_test[i])):
            ab = abs(Y_data_test[0][i][j] - y_predict[i][j])
            total_ground_truth += Y_data_test[0][i][j]
            total_mae += ab
            total_mse += (ab * ab)
            aa = math.log(Y_data_test[0][i][j] + 1, 2) - math.log(y_predict[i][j] + 1, 2)
            total_msle += (aa * aa)
            cc = abs(Y_data_test[0][i][j] - mean_gro)
            total_dis += (cc * cc)
            if Y_data_test[0][i][j] == 0:
                bb = 0
            else:
                bb = abs((Y_data_test[0][i][j] - y_predict[i][j]) / Y_data_test[0][i][j])
            total_mape += bb
            length_total += 1
    MAE_total_ = total_mae / length_total
    MSE_total_ = total_mse / length_total
    MSLE_total_ = total_msle / length_total
    MAPE_total_ = total_mape / length_total
    ER_total_ = total_mae / total_ground_truth
    R2_total_ = 1 - (total_mse / total_dis)

    print("City: " + city)
    print("MAE TOTAL: " + str(round(MAE_total_, 3)))
    print("MSE TOTAL: " + str(round(MSE_total_, 3)))
    print("RMSE TOTAL: " + str(round(math.sqrt(MSE_total_), 3)))
    print("MSLE TOTAL: " + str(round(MSLE_total_, 3)))
    print("MAPE TOTAL: " + str(round(MAPE_total_, 3)))
    print("Error Rate TOTAL: " + str(round(ER_total_, 3)))
    print("R-Squared TOTAL: " + str(R2_total_))
    print(str(MAE_total_) + " " + str(MSE_total_) + " " + str(ER_total_) + " " + str(MSLE_total_) + " " +
          str(MAPE_total_) + " " + str(math.sqrt(MSE_total_)) + " " + str(R2_total_))
    print("")

    running_secs = (dt.now() - start).seconds
    print("The running time is " + str(running_secs) + ' secs.')

    pyplot.plot(train_mse_all)  # label='train')
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Training MSE", fontsize=18)
    pyplot.legend()
    pyplot.show()

    pyplot.plot(val_mse_all)  # , label='train')
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("VAL MSE", fontsize=18)
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # tf.config.experimental_run_functions_eagerly(True)
    project_path = "D:\\ProgramProjects\\Python\\EALGAP\\"
    with tf.device("gpu:0"):
        extreme_high_valid_percent, extreme_low_valid_percent = 0.80, 0.20
        main("NYC_ClusterBasedHourlyPickUps_2020_20.h5", "NYC", project_path, extreme_high_valid_percent, extreme_low_valid_percent)