import tensorflow as tf
from tensorflow.python.keras.backend import flatten, batch_flatten
from tensorflow.python.keras.layers.ops.core import dense
from tensorflow.python.layers.base import Layer
from tensorflow.keras import initializers

class ExponentialDistributionBasedAttentionLayer(Layer):
    def __init__(self, len_input, node_num, f, d, decoder_layer_num, decoder_units):
        super(ExponentialDistributionBasedAttentionLayer, self).__init__(name="ExponentialDistributionBasedAttentionLayer")
        self.len_input = len_input
        self.f = f
        self.node_num = node_num
        self.d = d
        self.decoder_layer_num = decoder_layer_num
        self.decoder_units = decoder_units
        self.beta1 = self.add_weight(name="beta1", shape=tf.TensorShape([self.node_num, self.len_input]),
                                     dtype=tf.float32,
                                     initializer=initializers.random_normal(mean=0, stddev=1), trainable=True)
        self.gamma1 = self.add_weight(name="gamma1", shape=tf.TensorShape([self.node_num, self.len_input]),
                                      dtype=tf.float32,
                                      initializer=initializers.random_normal(mean=0, stddev=2), trainable=True)
        self.const = self.add_weight(name="const", shape=tf.TensorShape([1]), dtype=tf.float32,
                                     initializer=initializers.random_uniform(minval=0.01, maxval=2), trainable=True)


    '''
       data_input: (batch size, region_num/node_num, len_input, f)
       Qw: (batch size, region_num/node_num, f, d)
       Kw: (batch size, region_num/node_num, f, d)
       Vw: (batch size, region_num/node_num, f, d)
       output: (batch size, region_num/node_num, len_input, d)
    '''
    def exponential_distribution_based_temporal_attention_layer(self, data_input, pdf_input):
        extreme_probability = pdf_input
        Qw, Kw, Vw = self.decoder_layer(extreme_probability) # (batch size, region_num/node_num, f, d)
        Q = tf.matmul(data_input, Qw) # (batch size, region_num/node_num, len_input, d)
        K = tf.matmul(data_input, Kw) # (batch size, region_num/node_num, len_input, d)
        QK = tf.matmul(Q, K, transpose_b=True) # , transpose_b=True
        print("QK: " + str(QK))
        QK /= (self.d ** 0.5) # (batch size, region_num/node_num, len_input, len_input)
        attention_score = tf.nn.softmax(QK, axis=-1) # (batch size, region_num/node_num, len_input, len_input)
        print("attention_score: " + str(attention_score))
        V = tf.matmul(data_input, Vw) # (batch size, region_num/node_num, len_input, d)
        print("V: " + str(V))
        h = tf.matmul(attention_score, V) # (batch size, region_num/node_num, len_input, d)
        print("h: " + str(h))

        return h, Q, K, V, QK, data_input, attention_score, extreme_probability

    '''
        utilize the extreme_probability vector with length of len_input of each time series to generate the query, key 
        and value parameter vectors with shape of (batch size, f, d) for each region
        Qw: (batch size, region_num/node_num, f, d)
        Kw: (batch size, region_num/node_num, f, d)
        Vw: (batch size, region_num/node_num, f, d)
    '''
    def decoder_layer(self, extreme_probability):
        Qw = batch_flatten(extreme_probability)
        Kw = batch_flatten(extreme_probability)
        Vw = batch_flatten(extreme_probability)

        for i in range(self.decoder_layer_num):
            if i == 0:
                input_shape = self.node_num * self.len_input
            else:
                input_shape = self.decoder_units[i - 1]
            kernel_ = self.add_weight("Qw_kernel_" + str(i), shape=[input_shape, self.decoder_units[i]], trainable=True)
            Qw = dense(inputs=Qw, kernel=kernel_)
            # Qw = tf.nn.sigmoid(Qw)
            Qw = tf.nn.softmax(Qw)
            kernel_ = self.add_weight("Kw_kernel_" + str(i), shape=[input_shape, self.decoder_units[i]], trainable=True)
            Kw = dense(inputs=Kw, kernel=kernel_)
            # Kw = tf.nn.sigmoid(Kw)
            Kw = tf.nn.softmax(Kw)
            kernel_ = self.add_weight("Vw_kernel_" + str(i), shape=[input_shape, self.decoder_units[i]], trainable=True)
            Vw = dense(inputs=Vw, kernel=kernel_)
            # Vw = tf.nn.sigmoid(Vw)
            Vw = tf.nn.softmax(Vw)

        print("Qw: " + str(Qw))
        print("Kw: " + str(Kw))
        print("Vw: " + str(Vw))

        batch_size = tf.shape(extreme_probability)[0]
        Qw = tf.reshape(Qw, [batch_size, self.node_num, self.f, self.d])
        Kw = tf.reshape(Kw, [batch_size, self.node_num, self.f, self.d])
        Vw = tf.reshape(Vw, [batch_size, self.node_num, self.f, self.d])

        print("Qw: " + str(Qw))
        print("Kw: " + str(Kw))
        print("Vw: " + str(Vw))

        return Qw, Kw, Vw

    def normalization_layer(self, data_input, same_hour_mean_input, same_hour_var_input):
        batch_size = tf.shape(data_input)[0]
        data_input = tf.reshape(data_input, [batch_size, self.node_num, self.len_input])

        print("mean: " + str(same_hour_mean_input))
        print("var: " + str(same_hour_var_input))

        gamma = tf.tile([self.gamma1], [batch_size, 1, 1])
        print("gamma: " + str(gamma))
        beta = tf.tile([self.beta1], [batch_size, 1, 1])
        print("beta: " + str(beta))

        x_norm = tf.math.divide_no_nan(data_input - same_hour_mean_input, (same_hour_var_input + self.const) ** 0.5)
        out = tf.multiply(x_norm, gamma) + beta
        # out = tf.nn.tanh(out)
        out = tf.nn.softmax(out)

        return out