import tensorflow as tf
from tensorflow.python.keras.backend import flatten, batch_flatten
from tensorflow.python.keras.layers.ops.core import dense
from tensorflow.python.layers.base import Layer
from tensorflow.keras import initializers
from tensorflow.python.keras.layers import GRU

from Model.Layers.GEVDBasedAttentionLayer import ExponentialDistributionBasedAttentionLayer

'''
   Extract the extreme degree (both extreme high and extreme low) of  each time series in each time step
'''
class TemporalNormalizationLayer(Layer):
    def __init__(self, len_day, len_input, node_num, consider_his_data=True, momentum=0.1):
        super(TemporalNormalizationLayer, self).__init__(name="TemporalNormalizationLayer")
        self.len_day = len_day
        self.len_input = len_input
        self.node_num = node_num
        self.beta = self.add_weight(name="beta", shape=tf.TensorShape([self.node_num]), dtype=tf.float32,
                                    initializer=initializers.random_normal(mean=0, stddev=1), trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=tf.TensorShape([self.node_num]), dtype=tf.float32,
                                     initializer=initializers.random_normal(mean=0, stddev=1), trainable=True)
        self.running_mean = self.add_weight(name="running_mean", shape=tf.TensorShape([self.node_num]), dtype=tf.float32,
                                            initializer=initializers.random_normal(mean=0, stddev=1), trainable=True)
        self.running_var = self.add_weight(name="running_var", shape=tf.TensorShape([self.node_num]), dtype=tf.float32,
                                           initializer=initializers.random_uniform(minval=0.01, maxval=10), trainable=True)
        self.const = self.add_weight(name="const", shape=tf.TensorShape([1]), dtype=tf.float32,
                                     initializer=initializers.random_uniform(minval=0.01, maxval=2), trainable=True)
        self.momentum = momentum
        self.consider_his_data = consider_his_data
        self.beta1 = self.add_weight(name="beta1", shape=tf.TensorShape([self.node_num, self.len_input]), dtype=tf.float32,
                                     initializer=initializers.random_normal(mean=0, stddev=1), trainable=True)
        self.gamma1 = self.add_weight(name="gamma1", shape=tf.TensorShape([self.node_num, self.len_input]), dtype=tf.float32,
                                      initializer=initializers.random_normal(mean=0, stddev=2), trainable=True)

    '''
       data_input: (batch size, region_num/node_num, len_input)
       output: (batch size, region_num/node_num, len_input)
    '''
    def temporal_normalization_layer(self, data_input):
        batch_size = tf.shape(data_input)[0]
        if self.consider_his_data:
            n = self.node_num
            mean = tf.math.reduce_mean(data_input, axis=2) # shape=(None, self.node_num)
            var = tf.math.reduce_variance(data_input, axis=2) # shape=(None, self.node_num)
            print("reduced mean: " + str(mean))
            print("reduced var: " + str(var))
            mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
        else:
            mean = tf.math.reduce_mean(data_input, axis=2)
            var = tf.math.reduce_variance(data_input, axis=2)

        mean = tf.reshape(mean, [batch_size, self.node_num, 1])
        var = tf.reshape(var, [batch_size, self.node_num, 1])
        mean = tf.tile(mean, [1, 1, self.len_input])
        print("mean: " + str(mean))
        var = tf.tile(var, [1, 1, self.len_input])
        print("var: " + str(var))

        self.gamma = tf.reshape(self.gamma, [self.node_num, 1])
        gamma = tf.tile(self.gamma, [1, self.len_input])
        gamma = tf.tile([gamma], [batch_size, 1, 1])
        print("gamma: " + str(gamma))
        self.beta = tf.reshape(self.beta, [self.node_num, 1])
        beta = tf.tile(self.beta, [1, self.len_input])
        beta = tf.tile([beta], [batch_size, 1, 1])
        print("beta: " + str(beta))

        x_norm = tf.math.divide_no_nan(data_input - mean, (var + self.const) ** 0.5)
        out = tf.multiply(x_norm, gamma) + beta
        # tanh: scale the extreme degree to [-1,1]
        # softmax: scale the extreme degree to [0,1]
        # out = tf.nn.softmax(out)
        # out = tf.nn.tanh(out)
        out = tf.unstack(out, self.node_num, 1)
        new_out = []
        for i in range(0, self.node_num):
            new_out.append(tf.nn.tanh(out[i]))
            # new_out.append(tf.nn.softmax(out[i]))
        out = tf.stack(new_out, 1)

        return out

    def temporal_normalization_layer1(self, data_input):
        batch_size = tf.shape(data_input)[0]
        if self.consider_his_data:
            n = self.node_num
            mean = tf.math.reduce_mean(data_input, axis=2) # shape=(None, self.node_num)
            var = tf.math.reduce_variance(data_input, axis=2) # shape=(None, self.node_num)
            print("reduced mean: " + str(mean))
            print("reduced var: " + str(var))
            mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
        else:
            mean = tf.math.reduce_mean(data_input, axis=2)
            var = tf.math.reduce_variance(data_input, axis=2)

        mean = tf.reshape(mean, [batch_size, self.node_num, 1])
        var = tf.reshape(var, [batch_size, self.node_num, 1])
        mean = tf.tile(mean, [1, 1, self.len_input])
        print("mean: " + str(mean))
        var = tf.tile(var, [1, 1, self.len_input])
        print("var: " + str(var))

        gamma = tf.tile([self.gamma1], [batch_size, 1, 1])
        print("gamma: " + str(gamma))
        beta = tf.tile([self.beta1], [batch_size, 1, 1])
        print("beta: " + str(beta))

        x_norm = tf.math.divide_no_nan(data_input - mean, (var + self.const) ** 0.5)
        out = tf.multiply(x_norm, gamma)# + beta
        # tanh: scale the extreme degree to [-1,1]
        # softmax: scale the extreme degree to [0,1]
        # out = tf.nn.softmax(out)
        # out = tf.nn.tanh(out)
        out = tf.unstack(out, self.node_num, 1)
        new_out = []
        for i in range(0, self.node_num):
            new_out.append(tf.nn.tanh(out[i]))
            # new_out.append(tf.nn.softmax(out[i]))
        out = tf.stack(new_out, 1)

        return out, x_norm, mean, var, gamma, beta

    def temporal_normalization_layer2(self, data_input, same_hour_mean_input, same_hour_var_input):
        batch_size = tf.shape(data_input)[0]

        # print("mean: " + str(same_hour_mean_input))
        # print("var: " + str(same_hour_var_input))

        gamma = tf.tile([self.gamma1], [batch_size, 1, 1])
        # print("gamma: " + str(gamma))
        beta = tf.tile([self.beta1], [batch_size, 1, 1])
        # print("beta: " + str(beta))

        x_norm = tf.math.divide_no_nan(data_input - same_hour_mean_input, (same_hour_var_input + self.const) ** 0.5)
        out = tf.multiply(x_norm, gamma)# + beta
        # tanh: scale the extreme degree to [-1,1]
        # softmax: scale the extreme degree to [0,1]
        out = tf.unstack(out, self.node_num, 1)
        new_out = []
        for i in range(0, self.node_num):
            new_out.append(tf.nn.tanh(out[i]))
            # new_out.append(tf.nn.softmax(out[i]))
        out = tf.stack(new_out, 1)

        return out, x_norm, same_hour_mean_input, same_hour_var_input, gamma, beta

    def ground_truth_predicted_data_temporal_normalization(self, data_input, same_hour_mean_input, same_hour_var_input):
        out = tf.math.divide_no_nan(data_input - same_hour_mean_input, (same_hour_var_input) ** 0.5)
        # tanh: scale the extreme degree to [-1,1]
        # softmax: scale the extreme degree to [0,1]
        # out = tf.nn.softmax(out)
        out = tf.nn.tanh(out)

        return out


    '''
      The number of windows is len_day
    '''
    def window_based_temporal_normalization(self, window_data_X_input, window_same_hour_mean_X_input, window_same_hour_var_X_input):
        window_data_X_input = tf.unstack(window_data_X_input, self.len_day, 1)
        window_same_hour_mean_X_input = tf.unstack(window_same_hour_mean_X_input, self.len_day, 1)
        window_same_hour_var_X_input = tf.unstack(window_same_hour_var_X_input, self.len_day, 1)

        nor_window_data_X = []
        for i in range(self.len_day):
            out = self.temporal_normalization_layer2(window_data_X_input[i], window_same_hour_mean_X_input[i], window_same_hour_var_X_input[i])
            nor_window_data_X.append(out[0])
        nor_window_data_X = tf.stack(nor_window_data_X, 1)

        return nor_window_data_X

class ExtremeDegreeModelling(Layer):
    def __init__(self, len_day, node_num, len_input, f, d, decoder_units, decoder_layer_num):
        """
        len_day : number of windows in window_extreme_degree_input and window_predicted_extreme_degree_input
        len_input : length of input historical data of each region in each window data
        node_num : number of regions/nodes/sensors
        """
        super(ExtremeDegreeModelling, self).__init__(name="ExtremeDegreeModelling")
        self.len_day = len_day
        self.node_num = node_num
        self.len_input = len_input
        self.batch_size = None
        self.f = f
        self.d = d
        self.decoder_units = decoder_units
        self.decoder_layer_num = decoder_layer_num
        self.initial_state = None
        # use the tanh activation function to bound the scale of the predicted extreme degree into the range of [-1, 1]
        self.gru = GRU(self.node_num, return_state=True, activation="tanh")
        self.b = self.add_weight(name="b", shape=tf.TensorShape([self.node_num]),
                                 initializer=initializers.RandomNormal(mean=0, stddev=1), trainable=True)

        self.ed_attention = ExponentialDistributionBasedAttentionLayer(self.len_input, self.node_num, self.f, self.d,
                                                                       self.decoder_layer_num, self.decoder_units)

    '''
       Modeling the extreme degree patterns for each region of the city and predict the extreme degree of each region in the predicted time step
       window_extreme_degree_input: (len_day, len_input, node_num), the modeled window size extreme degree
       window_predicted_extreme_degree_input: (len_day, node_num), the ground truth extreme degree
       window_predicted_input: (len_day, node_num), the groundtruth data of all regions in each predicted time step of each window
       preliminary_pre: (node_num,), the preliminary prediction of node_num regions/nodes in the next following time step from other component
       output: (node_num,)
    '''
    def extreme_degree_modeling(self, window_extreme_degree_input, preliminary_pre):
        window_extreme_degree_input = tf.unstack(window_extreme_degree_input, self.len_day, 1)
        # print("window_extreme_degree_input: " + str(window_extreme_degree_input))

        all_h = []
        ct = []
        ct_sum = None
        for i in range(0, self.len_day):
            # utilize the historical modeled extreme degree of each region in one window to predict the extreme degree
            # in one following time step. The predicted time step is in the same hour of a day as the target time step of the whole model.
            pre_extreme_degree, last_state = self.gru(window_extreme_degree_input[i], initial_state=self.initial_state)
            # reset the initial_state so that the last output state from the gru in this round can be used as the input initial_state of next round
            self.reset_state(last_state)
            ctj = pre_extreme_degree
            ct.append(ctj)
            if ct_sum is None:
                ct_sum = ctj
            else:
                ct_sum = tf.add(ct_sum, ctj)
            all_h.append(pre_extreme_degree)
        output = preliminary_pre + tf.math.multiply(preliminary_pre, pre_extreme_degree)
        print("Output: " + str(output))

        return output, pre_extreme_degree

    def window_preliminary_pre(self, data_X, exponential_pms):
        gev_output, Q, K, V, QK, data_input, attention_score, extreme_probability = \
            self.ed_attention.exponential_distribution_based_temporal_attention_layer(data_X, exponential_pms)
        gev_output = tf.reshape(gev_output, [self.batch_size, self.node_num, self.len_input * self.d])

        return gev_output, Q, K, V, QK, data_input, attention_score, extreme_probability

    def reset_state(self, last_state):
        self.initial_state = last_state