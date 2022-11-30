from __future__ import print_function

import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Reshape, Dense, Permute, Activation
from tensorflow.python.keras.models import Model

from Model.Layers.GEVDBasedAttentionLayer import ExponentialDistributionBasedAttentionLayer
from Model.Layers.TemporalNormalizationLayer import TemporalNormalizationLayer, ExtremeDegreeModelling


def RegionalDataPredictionModel(node_num=10, len_closeness=5, len_day=3, T=24, embedding_length=5, d=1):
    main_inputs = []
    outputs = []

    category_input = Input(shape=(len_closeness, node_num))
    main_inputs.append(category_input)
    exponential_pms_input = Input(shape=(node_num, len_closeness))
    main_inputs.append(exponential_pms_input)
    same_hour_mean_input = Input(shape=(len_closeness, node_num))
    main_inputs.append(same_hour_mean_input)
    same_hour_var_input = Input(shape=(len_closeness, node_num))
    main_inputs.append(same_hour_var_input)

    window_same_hour_mean_X_input = Input(shape=(len_day, len_closeness, node_num))
    main_inputs.append(window_same_hour_mean_X_input)
    window_same_hour_var_X_input = Input(shape=(len_day, len_closeness, node_num))
    main_inputs.append(window_same_hour_var_X_input)
    window_data_X_input = Input(shape=(len_day, len_closeness, node_num))
    main_inputs.append(window_data_X_input)
    window_exponential_pms_input = Input(shape=(len_day, node_num, len_closeness))
    main_inputs.append(window_exponential_pms_input)

    gru_units, dense_units = 256, 128
    h_c_s = Reshape((len_closeness, node_num, 1))(category_input)
    h_c_s = Permute([2, 1, 3])(h_c_s)
    print("h_c_s: " + str(h_c_s))


    '''
    Global Impact Modeling Module
       f: number of time series in each region, 1 in this study
       d: the desired feature num in each region after the self-attention operation
    '''
    f = 1
    d = 1
    decoder_units = [256, 128, node_num * f * d]
    decoder_layer_num = len(np.array(decoder_units))
    ed_attention = ExponentialDistributionBasedAttentionLayer(len_closeness, node_num, f, d, decoder_layer_num, decoder_units)
    gev_output, Q, K, V, QK, data_input, attention_score, extreme_probability = \
        ed_attention.exponential_distribution_based_temporal_attention_layer(h_c_s, exponential_pms_input)
    gev_output = Reshape((node_num, len_closeness * d))(gev_output)

    pre = Dense(gru_units, input_shape=(node_num, len_closeness))(gev_output)
    print("pre: " + str(pre))
    pre = Dense(dense_units, input_shape=(node_num, gru_units), activation="relu")(pre)
    print("pre: " + str(pre))
    pre = Dense(1, input_shape=(node_num, dense_units), activation="relu")(pre)
    print("pre: " + str(pre))
    pre = Reshape((node_num,))(pre)
    print("Output: " + str(pre))

    '''
       Extreme Degree and Local Impact Modeling Module
    '''
    tnl = TemporalNormalizationLayer(len_day, len_closeness, node_num, consider_his_data=True, momentum=0.2)
    window_data_X = Permute([1, 3, 2])(window_data_X_input)
    window_same_hour_mean_X = Permute([1, 3, 2])(window_same_hour_mean_X_input)
    window_same_hour_var_X = Permute([1, 3, 2])(window_same_hour_var_X_input)
    nor_window_data_X = tnl.window_based_temporal_normalization(window_data_X, window_same_hour_mean_X, window_same_hour_var_X)
    nor_window_data_X = Permute([1, 3, 2])(nor_window_data_X)
    edm = ExtremeDegreeModelling(len_day, node_num, len_closeness, f, d, decoder_units, decoder_layer_num)
    pre, pre_extreme_degree = edm.extreme_degree_modeling(nor_window_data_X, pre)
    pre = Activation(activation="relu")(pre)
    print("tnl_output: " + str(pre))

    outputs.append(pre)
    outputs.append(Q)
    outputs.append(K)
    outputs.append(V)
    outputs.append(QK)
    outputs.append(data_input)
    outputs.append(attention_score)
    outputs.append(extreme_probability)
    outputs.append(pre_extreme_degree)
    model = Model(main_inputs, outputs)

    return model