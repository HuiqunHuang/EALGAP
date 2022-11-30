from DataProcessing.PearsonCorrelation import Pearson_Correlation_Coefficient
from FileOperation.LoadH5 import loadDataDate
import numpy as np
import os
from config import Config
DATAPATH = Config().DATAPATH
import openturns as ot
ot.Log.Show(ot.Log.NONE)
ot.RandomGenerator.SetSeed(0)

from DataProcessing.TimeStrProcessing import weekend_2020

def loadDataForModel_NYC(datatype, data_file_name, T=24, len_closeness=4, len_trend=6, len_distribution=48, pdf_ratio=1000, len_day=7, len_test=None, len_val=None, extreme_high=None):
    assert (len_closeness > 0)

    if datatype == "nyc bike 2020":
        cluster_based_bike_data, time_data = loadDataDate(os.path.join(DATAPATH, 'Data\\NYC\\', data_file_name))


    if datatype == "nyc bike 2020":
        # using the data from 2020-05-09 to 2020-08-09
        cluster_based_bike_data = cluster_based_bike_data[T * 160:T * 222]
        # using the data from 2020-05-31 to 2020-08-31
        # cluster_based_bike_data = cluster_based_bike_data[T * 151:T * 244]
        # using the data from Oct. 01 to Dec. 31 2020
        # cluster_based_bike_data = cluster_based_bike_data[T * 274:]
        # using the data from Jue. 27 to Sep. 27 2020
        # cluster_based_bike_data = cluster_based_bike_data[178 * T:T * 271]

    extreme_label = []
    for i in range(0, len(cluster_based_bike_data)):
        label = []
        for j in range(0, len(cluster_based_bike_data[i])):
            if int(cluster_based_bike_data[i][j]) >= extreme_high:
                label.append(1)
            else:
                label.append(0)
        extreme_label.append(label)
    extreme_label = np.array(extreme_label)
    print("cluster_based_bike_data shape: " + str(cluster_based_bike_data.shape))
    print("extreme_label shape: " + str(extreme_label.shape))
    node_num = cluster_based_bike_data.shape[1]

    label_X, label_Y = [], []
    target_label_Y = []
    extreme_data_X, extreme_data_Y = [], []
    start_index = max([len_closeness, len_distribution, T * (len_day + 2), 7 * T * len_trend, int(len_closeness + 2 * ((len_day / 2) * T * 7 + (1 - len_day % 2) * T - 1))])
    print("start_index shape: " + str(start_index))
    max_same_hour = int((len_day / 2) * T * 7 + (1 - len_day % 2) * T)
    window = T * len_day
    near_category_X, near_category_y = [], []
    exponential_data_X = []
    same_hour_mean_X, same_hour_var_X = [], []
    same_hour_mean_Y, same_hour_var_Y = [], []
    window_same_hour_mean_X, window_same_hour_mean_Y = [], []
    window_same_hour_var_X, window_same_hour_var_Y = [], []
    window_data_X, window_data_Y, window_exponential_data_X = [], [], []
    spearson_data_X, window_spearson_data_X = [], []

    for i in range(start_index, len(cluster_based_bike_data)):
        near_category_X.append(
            np.asarray([np.vstack([cluster_based_bike_data[j] for j in range(i - len_closeness, i)])]))
        near_category_y.append(np.asarray(cluster_based_bike_data[i]))

        dd = []
        exponential_data_one = []
        dd_clo = []
        for j in range(0, len(cluster_based_bike_data[i])):
            dd.append([])
            dd_clo.append([])
        for j in range(i - len_distribution, i):
            for k in range(0, len(cluster_based_bike_data[j])):
                dd[k].append([cluster_based_bike_data[j][k]])
        for j in range(i - len_closeness, i):
            for k in range(0, len(cluster_based_bike_data[j])):
                dd_clo[k].append(cluster_based_bike_data[j][k])

        spearson_data_one = []
        for j in range(node_num):
            one = []
            for k in range(node_num):
                if j == k:
                    one.append(1)
                else:
                    one.append(0)
            spearson_data_one.append(one)
        for j in range(node_num):
            for k in range(j + 1, node_num):
                r, p = Pearson_Correlation_Coefficient(dd_clo[j], dd_clo[k])
                spearson_data_one[j][k] = r
                spearson_data_one[k][j] = r
        # print(spearson_data_one)
        spearson_data_X.append(np.asarray([spearson_data_one]))

        for j in range(0, len(dd_clo)):
            mean_one = np.mean(dd[j])
            one = []
            for k in range(0, len(dd_clo[j])):
                if dd_clo[j][k] >= 0 and mean_one != 0:
                    a = (1 / mean_one) * np.exp((-1 / mean_one) * dd_clo[j][k]) * pdf_ratio
                elif mean_one == 0 and dd_clo[j][k] == 0:
                    a = 1
                else:
                    a = 0
                one.append(a)
            # P = ss.norm.fit(dd_clo[j])
            # one = ss.norm.pdf(dd_clo[j], *P)
            exponential_data_one.append(one)
        # print(np.array(paisson_data_one).shape)
        exponential_data_X.append(np.asarray([exponential_data_one]))
        target_label_Y.append(np.asarray([extreme_label[i]]))

        # if dynamic extreme label
        x, y, label_x, label_y = [], [], [], []
        valid_count = 0
        for j in range(i, 0, -T):
            if valid_count < len_day:
                x.append(cluster_based_bike_data[j - 2 * T:j - T])
                y.append(cluster_based_bike_data[j - T])

                v = cluster_based_bike_data[j - 2 * T:j - T + 1]
                mean_value = np.mean(v)
                std_value = np.std(v)
                extreme_high_value = mean_value + std_value
                current_label = []
                for k in range(0, len(v)):
                    cl = []
                    for m in range(0, len(v[k])):
                        if v[k][m] >= extreme_high_value:
                            cl.append(1)
                        else:
                            cl.append(0)
                    current_label.append(cl)
                current_label = np.array(current_label)
                label_x.append(current_label[:-1])
                label_y.append(current_label[-1:][0])
                valid_count += 1
            if valid_count >= len_day:
                break
        if valid_count < len_day:
            print("Not enough data, please adjust start_index " + str(i))

        extreme_data_X.append([x])
        extreme_data_Y.append([y])
        label_X.append([label_x])
        label_Y.append([label_y])

        mean_one, var_one = [], []
        for j in range(i - len_closeness, i):
            if datatype == "nyc bike 2020":
                ifweekday = weekend_2020[j // T]

            one = []
            for k in range(node_num):
                one.append([])
            # for k in range(j - len_day * T, j, T):
            valid_days = 0
            for k in range(j, j - max_same_hour - 1, -T):
                if datatype == "nyc bike 2020":
                    ifweekday_one = weekend_2020[k // T]

                if ifweekday == ifweekday_one and valid_days < len_day:
                    for m in range(0, len(cluster_based_bike_data[k])):
                        one[m].append(cluster_based_bike_data[k][m])
                    valid_days += 1
                if valid_days >= len_day:
                    break
            mean_one.append(np.mean(np.array(one), axis=1))
            var_one.append(np.var(np.array(one), axis=1))
        same_hour_mean_X.append(np.asarray([np.vstack(mean_one)]))
        same_hour_var_X.append(np.asarray([np.vstack(var_one)]))

        one = []
        for j in range(node_num):
            one.append([])
        valid_days = 0
        for j in range(i, i - max_same_hour - 1, -T):
            if datatype == "nyc bike 2020":
                ifweekday_one = weekend_2020[j // T]

            if ifweekday == ifweekday_one and valid_days < len_day:
                for k in range(0, len(cluster_based_bike_data[j])):
                    one[k].append(cluster_based_bike_data[j][k])
                valid_days += 1
            if valid_days >= len_day:
                break
        mean_one.append(np.mean(np.array(one), axis=1))
        var_one.append(np.var(np.array(one), axis=1))
        same_hour_mean_Y.append(np.asarray(mean_one))
        same_hour_var_Y.append(np.asarray(var_one))


        # the length of the windows is len_day
        # x/y: the original data of len_day windows
        # mean_x/var_x/mean_y/var_y: the mean/var of each time step of len_day windows
        # x/mean_x/var_x: (len_day, len_closeness, node_num)
        # y/mean_y/var_y: (len_day, node_num)
        x, y, mean_x, mean_y, var_x, var_y = [], [], [], [], [], []
        exponential_data_x_one = []
        valid_count = 0
        if datatype == "nyc bike 2020":
            ifweekday = weekend_2020[i // T]

        window_spearson_data_X_one = []
        # process the data of len_day windows
        for j in range(i, 0, -T): # j is the index of the historical data at the same time of a day as the predicted time step, including the predicted time step
            if datatype == "nyc bike 2020":
                ifweekday_one = weekend_2020[j // T]

            # processing data of one window
            if ifweekday == ifweekday_one and valid_count < len_day: # if the historical data are also at weekday/weekend as the predicted time step
                x.append(cluster_based_bike_data[j - len_closeness:j]) # the original data of one time window
                y.append(cluster_based_bike_data[j])

                dd = []
                exponential_one = []
                dd_clo = []
                for k in range(0, len(cluster_based_bike_data[j])):
                    dd.append([])
                    dd_clo.append([])
                for k in range(j - len_distribution, j):
                    for m in range(0, len(cluster_based_bike_data[k])):
                        dd[m].append([cluster_based_bike_data[k][m]])
                for k in range(j - len_closeness, j):
                    for m in range(0, len(cluster_based_bike_data[k])):
                        dd_clo[m].append(cluster_based_bike_data[k][m])

                window_spearson_data_X_one_ = []
                for k in range(node_num):
                    one = []
                    for m in range(node_num):
                        if k == m:
                            one.append(1)
                        else:
                            one.append(0)
                    window_spearson_data_X_one_.append(one)
                for k in range(node_num):
                    for m in range(k + 1, node_num):
                        r, p = Pearson_Correlation_Coefficient(dd_clo[k], dd_clo[m])
                        window_spearson_data_X_one_[k][m] = r
                        window_spearson_data_X_one_[m][k] = r
                window_spearson_data_X_one.append(window_spearson_data_X_one_)

                for k in range(0, len(dd_clo)):
                    mean_one = np.mean(dd[k])
                    one = []
                    for m in range(0, len(dd_clo[k])):
                        if dd_clo[k][m] >= 0 and mean_one != 0:
                            a = (1 / mean_one) * np.exp((-1 / mean_one) * dd_clo[k][m]) * pdf_ratio
                        elif mean_one == 0 and dd_clo[k][m] == 0:
                            a = 1
                        else:
                            a = 0
                        one.append(a)
                    exponential_one.append(one)
                # print(np.array(paisson_data_one).shape)
                exponential_data_x_one.append(np.asarray(exponential_one))


                # calculate the mean and var for the data in each tme step within one window
                # mean_x_one: (len_closeness, node_num), mean_y_one: (node_num,)
                # var_x_one: (len_closeness, node_num), var_y_one: (node_num,)
                mean_x_one = []
                var_x_one = []
                for k in range(j - len_closeness, j):
                    if datatype == "nyc bike 2020":
                        ifw = weekend_2020[k // T]

                    valid_days = 0
                    one = []
                    for k in range(node_num):
                        one.append([])
                    for m in range(k, k - max_same_hour - 1, -T):
                        if datatype == "nyc bike 2020":
                            ifweekday_two = weekend_2020[m // T]

                        if ifw == ifweekday_two and valid_days < len_day:
                            for n in range(0, len(cluster_based_bike_data[m])):
                                one[n].append(cluster_based_bike_data[m][n])
                            valid_days += 1
                        # already processed the data in one specific time step within one window
                        if valid_days >= len_day:
                            break
                    mean_x_one.append(np.mean(np.array(one), axis=1))
                    var_x_one.append(np.var(np.array(one), axis=1))
                # print("mean_x_one: " + str(np.array(mean_x_one).shape)) # (len_closeness, node_num)
                valid_days = 0
                one = []
                for k in range(node_num):
                    one.append([])
                for k in range(j,  j - max_same_hour - 1, -T):
                    if datatype == "nyc bike 2020":
                        ifweekday_two = weekend_2020[k // T]

                    if ifweekday_one == ifweekday_two and valid_days < len_day:
                        for m in range(0, len(cluster_based_bike_data[k])):
                            one[m].append(cluster_based_bike_data[k][m])
                        valid_days += 1
                    if valid_days >= len_day:
                        break
                # print("one: " + str(np.array(one).shape))  # (node_num, len_day)
                mean_y.append(np.mean(np.array(one), axis=1))
                var_y.append(np.var(np.array(one), axis=1))
                mean_x.append(mean_x_one)
                var_x.append(var_x_one)
                valid_count += 1
            # already processed the data of all windows
            if valid_count >= len_day:
                break
        window_same_hour_mean_X.append(np.asarray([mean_x]))
        window_same_hour_var_X.append(np.asarray([var_x]))
        window_same_hour_mean_Y.append(np.asarray([mean_y]))
        window_same_hour_var_Y.append(np.asarray([var_y]))
        window_data_X.append(np.asarray([x]))
        window_data_Y.append(np.asarray([y]))
        window_exponential_data_X.append(np.asarray([exponential_data_x_one]))
        window_spearson_data_X.append(np.asarray([window_spearson_data_X_one]))

    label_X = np.vstack(label_X)
    label_Y = np.vstack(label_Y)
    target_label_Y = np.vstack(target_label_Y)
    extreme_data_X = np.vstack(extreme_data_X)
    extreme_data_Y = np.vstack(extreme_data_Y)
    near_category_X = np.vstack(near_category_X)
    near_category_y = np.vstack(near_category_y)
    exponential_data_X = np.vstack(exponential_data_X)
    same_hour_mean_X = np.vstack(same_hour_mean_X)
    same_hour_var_X = np.vstack(same_hour_var_X)

    window_same_hour_mean_X = np.vstack(window_same_hour_mean_X)
    window_same_hour_var_X = np.vstack(window_same_hour_var_X)
    window_same_hour_mean_Y = np.vstack(window_same_hour_mean_Y)
    window_same_hour_var_Y = np.vstack(window_same_hour_var_Y)
    window_data_X = np.vstack(window_data_X)
    window_data_Y = np.vstack(window_data_Y)
    window_exponential_data_X = np.vstack(window_exponential_data_X)
    same_hour_mean_Y = np.vstack(same_hour_mean_Y)
    same_hour_var_Y = np.vstack(same_hour_var_Y)

    window_spearson_data_X = np.vstack(window_spearson_data_X)
    spearson_data_X = np.vstack(spearson_data_X)

    print("near_category_X shape: " + str(near_category_X.shape))
    print("near_category_y shape: " + str(near_category_y.shape))
    print("label_X shape: " + str(label_X.shape))
    print("label_Y shape: " + str(label_Y.shape))
    print("extreme_data_X shape: " + str(extreme_data_X.shape))
    print("extreme_data_Y shape: " + str(extreme_data_Y.shape))
    print("target_label_Y shape: " + str(target_label_Y.shape))
    print("exponential_data_X shape: " + str(exponential_data_X.shape))
    print("same_hour_mean_X shape: " + str(same_hour_mean_X.shape))
    print("same_hour_var_X shape: " + str(same_hour_var_X.shape))
    print("same_hour_mean_Y shape: " + str(same_hour_mean_Y.shape))
    print("same_hour_var_Y shape: " + str(same_hour_var_Y.shape))
    print("window_same_hour_mean_X shape: " + str(window_same_hour_mean_X.shape))
    print("window_same_hour_var_X shape: " + str(window_same_hour_var_X.shape))
    print("window_same_hour_mean_Y shape: " + str(window_same_hour_mean_Y.shape))
    print("window_same_hour_var_Y shape: " + str(window_same_hour_var_Y.shape))
    print("window_data_X shape: " + str(window_data_X.shape))
    print("window_data_Y shape: " + str(window_data_Y.shape))
    print("window_exponential_data_X shape: " + str(window_exponential_data_X.shape))
    print("window_spearson_data_X shape: " + str(window_spearson_data_X.shape))
    print("spearson_data_X shape: " + str(spearson_data_X.shape))


    exponential_data_X_train, exponential_data_X_val, exponential_data_X_test = \
        exponential_data_X[:-(len_test + len_val)], exponential_data_X[-(len_test + len_val):-len_test],\
        exponential_data_X[-len_test:]

    near_category_X_train, near_category_X_val, near_category_X_test = \
        near_category_X[:-(len_test + len_val)], near_category_X[-(len_test + len_val):-len_test], near_category_X[
                                                                                                   -len_test:]
    near_category_y_train, near_category_y_val, near_category_y_test = \
        near_category_y[:-(len_test + len_val)], near_category_y[-(len_test + len_val):-len_test], near_category_y[
                                                                                                   -len_test:]

    label_X_train, label_X_val, label_X_test = label_X[:-(len_test + len_val)], label_X[-(
            len_test + len_val):-len_test], label_X[-len_test:]
    label_Y_train, label_Y_val, label_Y_test = label_Y[:-(len_test + len_val)], label_Y[-(
            len_test + len_val):-len_test], label_Y[-len_test:]
    extreme_data_X_train, extreme_data_X_val, extreme_data_X_test = \
        extreme_data_X[:-(len_test + len_val)], extreme_data_X[-(len_test + len_val):-len_test], extreme_data_X[
                                                                                                 -len_test:]
    extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test = \
        extreme_data_Y[:-(len_test + len_val)], extreme_data_Y[-(len_test + len_val):-len_test], extreme_data_Y[
                                                                                                 -len_test:]
    target_label_Y_train, target_label_Y_val, target_label_Y_test = \
        target_label_Y[:-(len_test + len_val)], target_label_Y[-(len_test + len_val):-len_test], target_label_Y[
                                                                                                 -len_test:]
    same_hour_mean_X_train, same_hour_mean_X_val, same_hour_mean_X_test = \
        same_hour_mean_X[:-(len_test + len_val)], same_hour_mean_X[-(len_test + len_val):-len_test], same_hour_mean_X[
                                                                                                     -len_test:]
    same_hour_var_X_train, same_hour_var_X_val, same_hour_var_X_test = \
        same_hour_var_X[:-(len_test + len_val)], same_hour_var_X[-(len_test + len_val):-len_test], same_hour_var_X[
                                                                                                   -len_test:]

    same_hour_mean_Y_train, same_hour_mean_Y_val, same_hour_mean_Y_test = \
        same_hour_mean_Y[:-(len_test + len_val)], same_hour_mean_Y[-(len_test + len_val):-len_test], same_hour_mean_Y[
                                                                                                     -len_test:]
    same_hour_var_Y_train, same_hour_var_Y_val, same_hour_var_Y_test = \
        same_hour_var_Y[:-(len_test + len_val)], same_hour_var_Y[-(len_test + len_val):-len_test], same_hour_var_Y[
                                                                                                   -len_test:]

    window_same_hour_mean_X_train, window_same_hour_mean_X_val, window_same_hour_mean_X_test = \
        window_same_hour_mean_X[:-(len_test + len_val)], window_same_hour_mean_X[-(len_test + len_val):-len_test], \
        window_same_hour_mean_X[-len_test:]
    window_same_hour_mean_Y_train, window_same_hour_mean_Y_val, window_same_hour_mean_Y_test = \
        window_same_hour_mean_Y[:-(len_test + len_val)], window_same_hour_mean_Y[-(len_test + len_val):-len_test], \
        window_same_hour_mean_Y[-len_test:]
    window_same_hour_var_X_train, window_same_hour_var_X_val, window_same_hour_var_X_test = \
        window_same_hour_var_X[:-(len_test + len_val)], window_same_hour_var_X[-(len_test + len_val):-len_test], \
        window_same_hour_var_X[-len_test:]
    window_same_hour_var_Y_train, window_same_hour_var_Y_val, window_same_hour_var_Y_test = \
        window_same_hour_var_Y[:-(len_test + len_val)], window_same_hour_var_Y[-(len_test + len_val):-len_test], \
        window_same_hour_var_Y[-len_test:]
    window_data_X_train, window_data_X_val, window_data_X_test = \
        window_data_X[:-(len_test + len_val)], window_data_X[-(len_test + len_val):-len_test], \
        window_data_X[-len_test:]
    window_data_Y_train, window_data_Y_val, window_data_Y_test = \
        window_data_Y[:-(len_test + len_val)], window_data_Y[-(len_test + len_val):-len_test], \
        window_data_Y[-len_test:]

    window_exponential_data_X_train, window_exponential_data_X_val, window_exponential_data_X_test = \
        window_exponential_data_X[:-(len_test + len_val)], window_exponential_data_X[-(len_test + len_val):-len_test], \
        window_exponential_data_X[-len_test:]

    window_spearson_data_X_train, window_spearson_data_X_val, window_spearson_data_X_test = \
        window_spearson_data_X[:-(len_test + len_val)], window_spearson_data_X[-(len_test + len_val):-len_test], \
        window_spearson_data_X[-len_test:]
    spearson_data_X_train, spearson_data_X_val, spearson_data_X_test = \
        spearson_data_X[:-(len_test + len_val)], spearson_data_X[-(len_test + len_val):-len_test], \
        spearson_data_X[-len_test:]

    print("near_category_X_train shape: " + str(near_category_X_train.shape))
    print("near_category_X_val shape: " + str(near_category_X_val.shape))
    print("near_category_X_test shape: " + str(near_category_X_test.shape))
    print("")
    print("near_category_y_train shape: " + str(near_category_y_train.shape))
    print("near_category_y_val shape: " + str(near_category_y_val.shape))
    print("near_category_y_test shape: " + str(near_category_y_test.shape))
    print("")
    print("label_X_train shape: " + str(label_X_train.shape))
    print("label_X_val shape: " + str(label_X_val.shape))
    print("label_X_test shape: " + str(label_X_test.shape))
    print("")
    print("extreme_data_X_train shape: " + str(extreme_data_X_train.shape))
    print("extreme_data_X_val shape: " + str(extreme_data_X_val.shape))
    print("extreme_data_X_test shape: " + str(extreme_data_X_test.shape))
    print("")
    print("label_Y_train shape: " + str(label_Y_train.shape))
    print("label_Y_val shape: " + str(label_Y_val.shape))
    print("label_Y_test shape: " + str(label_Y_test.shape))
    print("")
    print("target_label_Y_train shape: " + str(target_label_Y_train.shape))
    print("target_label_Y_val shape: " + str(target_label_Y_val.shape))
    print("target_label_Y_test shape: " + str(target_label_Y_test.shape))
    print("")
    print("exponential_data_X_train shape: " + str(exponential_data_X_train.shape))
    print("exponential_data_X_val shape: " + str(exponential_data_X_val.shape))
    print("exponential_data_X_test shape: " + str(exponential_data_X_test.shape))
    print("")
    print("same_hour_mean_X_train shape: " + str(same_hour_mean_X_train.shape))
    print("same_hour_mean_X_val shape: " + str(same_hour_mean_X_val.shape))
    print("same_hour_mean_X_test shape: " + str(same_hour_mean_X_test.shape))
    print("")
    print("same_hour_var_X_train shape: " + str(same_hour_var_X_train.shape))
    print("same_hour_var_X_val shape: " + str(same_hour_var_X_val.shape))
    print("same_hour_var_X_test shape: " + str(same_hour_var_X_test.shape))
    print("")
    print("same_hour_mean_Y_train shape: " + str(same_hour_mean_Y_train.shape))
    print("same_hour_mean_Y_val shape: " + str(same_hour_mean_Y_val.shape))
    print("same_hour_mean_Y_test shape: " + str(same_hour_mean_Y_test.shape))
    print("")
    print("same_hour_var_Y_train shape: " + str(same_hour_var_Y_train.shape))
    print("same_hour_var_Y_val shape: " + str(same_hour_var_Y_val.shape))
    print("same_hour_var_Y_test shape: " + str(same_hour_var_Y_test.shape))
    print("")
    print("window_same_hour_mean_X_train shape: " + str(window_same_hour_mean_X_train.shape))
    print("window_same_hour_mean_X_val shape: " + str(window_same_hour_mean_X_val.shape))
    print("window_same_hour_mean_X_test shape: " + str(window_same_hour_mean_X_test.shape))
    print("")
    print("window_same_hour_mean_Y_train shape: " + str(window_same_hour_mean_Y_train.shape))
    print("window_same_hour_mean_Y_val shape: " + str(window_same_hour_mean_Y_val.shape))
    print("window_same_hour_mean_Y_test shape: " + str(window_same_hour_mean_Y_test.shape))
    print("")

    print("window_same_hour_var_X_train shape: " + str(window_same_hour_var_X_train.shape))
    print("window_same_hour_var_X_val shape: " + str(window_same_hour_var_X_val.shape))
    print("window_same_hour_var_X_test shape: " + str(window_same_hour_var_X_test.shape))
    print("")
    print("window_same_hour_var_Y_train shape: " + str(window_same_hour_var_Y_train.shape))
    print("window_same_hour_var_Y_val shape: " + str(window_same_hour_var_Y_val.shape))
    print("window_same_hour_var_Y_test shape: " + str(window_same_hour_var_Y_test.shape))
    print("")

    print("window_data_X_train shape: " + str(window_data_X_train.shape))
    print("window_data_X_val shape: " + str(window_data_X_val.shape))
    print("window_data_X_test shape: " + str(window_data_X_test.shape))
    print("")
    print("window_data_Y_train shape: " + str(window_data_Y_train.shape))
    print("window_data_Y_val shape: " + str(window_data_Y_val.shape))
    print("window_data_Y_test shape: " + str(window_data_Y_test.shape))
    print("")
    print("window_exponential_data_X_train shape: " + str(window_exponential_data_X_train.shape))
    print("window_exponential_data_X_val shape: " + str(window_exponential_data_X_val.shape))
    print("window_exponential_data_X_test shape: " + str(window_exponential_data_X_test.shape))
    print("")

    print("window_spearson_data_X_train shape: " + str(window_spearson_data_X_train.shape))
    print("window_spearson_data_X_val shape: " + str(window_spearson_data_X_val.shape))
    print("window_spearson_data_X_test shape: " + str(window_spearson_data_X_test.shape))
    print("")
    print("spearson_data_X_train shape: " + str(spearson_data_X_train.shape))
    print("spearson_data_X_val shape: " + str(spearson_data_X_val.shape))
    print("spearson_data_X_test shape: " + str(spearson_data_X_test.shape))
    print("")


    X_data_train = []
    X_data_val = []
    X_data_test = []
    for l, X_ in zip([len_closeness, len_closeness, len_closeness, len_closeness, len_day, len_day, len_day, len_day, 1, len_day
                      ], [near_category_X_train, exponential_data_X_train, same_hour_mean_X_train,
                          same_hour_var_X_train, window_same_hour_mean_X_train, window_same_hour_var_X_train,
                          window_data_X_train, window_exponential_data_X_train, spearson_data_X_train,
                          window_spearson_data_X_train]):
        if l > 0:
            X_data_train.append(X_)
    for l, X_ in zip([len_closeness, len_closeness, len_closeness, len_closeness, len_day, len_day, len_day, len_day, 1, len_day
                      ], [near_category_X_val, exponential_data_X_val, same_hour_mean_X_val, same_hour_var_X_val,
                          window_same_hour_mean_X_val, window_same_hour_var_X_val, window_data_X_val,
                          window_exponential_data_X_val, spearson_data_X_val, window_spearson_data_X_val]):
        if l > 0:
            X_data_val.append(X_)
    for l, X_ in zip([len_closeness, len_closeness, len_closeness, len_closeness, len_day, len_day, len_day, len_day, 1, len_day
                      ], [near_category_X_test, exponential_data_X_test, same_hour_mean_X_test, same_hour_var_X_test,
                          window_same_hour_mean_X_test, window_same_hour_var_X_test, window_data_X_test,
                          window_exponential_data_X_test, spearson_data_X_test, window_spearson_data_X_test]):
        if l > 0:
            X_data_test.append(X_)
    Y_data_train = []
    Y_data_val = []
    Y_data_test = []
    for l, X_ in zip([1, 1], [near_category_y_train, window_data_Y_train]):
        if l > 0:
            Y_data_train.append(X_)
    for l, X_ in zip([1, 1], [near_category_y_val, window_data_Y_val]):
        if l > 0:
            Y_data_val.append(X_)
    for l, X_ in zip([1], [near_category_y_test, window_data_Y_test]):
        if l > 0:
            Y_data_test.append(X_)

    for _X in X_data_train:
        print(np.array(_X).shape, )
    print()
    for _X in X_data_val:
        print(np.array(_X).shape, )
    print()
    for _X in X_data_test:
        print(np.array(_X).shape, )
    print()

    return X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, \
           np.array(near_category_X_train), np.array(near_category_X_val), np.array(near_category_X_test), \
           np.array(near_category_y_train), np.array(near_category_y_val), np.array(near_category_y_test), \
           extreme_data_X_train, extreme_data_X_val, extreme_data_X_test, \
           extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test, \
           label_Y_train, label_Y_val, label_Y_test, \
           np.array(target_label_Y_train), target_label_Y_val, target_label_Y_test, \
           np.array(exponential_data_X_train), np.array(exponential_data_X_val), np.array(exponential_data_X_test), \
           np.array(same_hour_mean_X_train), np.array(same_hour_mean_X_val), np.array(same_hour_mean_X_test), \
           np.array(same_hour_var_X_train), np.array(same_hour_var_X_val), np.array(same_hour_var_X_test), \
           np.array(same_hour_mean_Y_train), np.array(same_hour_mean_Y_val), np.array(same_hour_mean_Y_test), \
           np.array(same_hour_var_Y_train), np.array(same_hour_var_Y_val), np.array(same_hour_var_Y_test), \
           window_same_hour_mean_X_train, window_same_hour_mean_X_val, window_same_hour_mean_X_test, \
           window_same_hour_mean_Y_train, window_same_hour_mean_Y_val, window_same_hour_mean_Y_test, \
           window_same_hour_var_X_train, window_same_hour_var_X_val, window_same_hour_var_X_test, \
           window_same_hour_var_Y_train, window_same_hour_var_Y_val, window_same_hour_var_Y_test, \
           window_data_X_train, window_data_X_val, window_data_X_test, \
           window_data_Y_train, window_data_Y_val, window_data_Y_test, \
           window_exponential_data_X_train, window_exponential_data_X_val, window_exponential_data_X_test, \
           spearson_data_X_train, spearson_data_X_val, spearson_data_X_test, \
           window_spearson_data_X_train, window_spearson_data_X_val, window_spearson_data_X_test
