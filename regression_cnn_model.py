# coding: utf-8

import pickle

import numpy as np
import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 300  # 词向量维度
    seq_length = 1500  # 序列长度
    num_classes = -1  # 类别数
    num_filters = 100  # 卷积核数目
    kernel_size = 2  # 卷积核尺寸
    vocab_size = -1  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 1  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 100  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    Use_tags = False  # 是否使用标签信息
    Three_filter_open = False  # 3种卷积核大小模式
    Use_embedding = False  # 使用word2vec
    choose_wordVector = 0  # 0是glove,1是word2vector
    Use_batch_normalization = False  # 使用BN

    num_hidden_layers = 2  # 隐藏层数量


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.input_re = tf.placeholder(tf.float32, [None, 6], name='input_re')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.training = tf.placeholder(tf.bool, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # self.input_y = self.input_y/10
        # 词向量加载
        if self.config.Use_embedding:
            with open('data/' + str(self.config.num_classes) + "/word_vector.pkl", 'rb') as f:
                embedding_weights = pickle.load(f)

        # 词向量映射
        with tf.device('/cpu:0'):
            if self.config.Use_embedding:
                embedding = tf.Variable(np.array(embedding_weights), trainable=True, name='embedding_weights',
                                        dtype='float32')
            else:
                embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):

            regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
            # CNN layer
            conv_0 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv_0'
                                      )
            if self.config.Three_filter_open:
                conv_1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size - 1,
                                          name='conv_1')
                conv_2 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + 1,
                                          name='conv_2')

            # global max pooling layer

            gmp_0 = tf.reduce_max(conv_0, reduction_indices=[1], name='gmp_0')
            if self.config.Three_filter_open:
                gmp_1 = tf.reduce_max(conv_1, reduction_indices=[1], name='gmp_1')
                gmp_2 = tf.reduce_max(conv_2, reduction_indices=[1], name='gmp_2')

                gmp_all = tf.concat([gmp_0, gmp_1, gmp_2], 1, name='combine')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            if self.config.Three_filter_open:
                if self.config.Use_batch_normalization:
                    gmp_all = tf.layers.batch_normalization(gmp_all, training=self.training, momentum=0.9)
                gmp_out = gmp_all
            else:
                if self.config.Use_batch_normalization:
                    gmp_0 = tf.layers.batch_normalization(gmp_0, training=self.training, momentum=0.9)
                gmp_out = gmp_0
            gmp_out = tf.concat([gmp_out, self.input_re], 1, name='combine__')
            if self.config.Use_tags:
                fc_list = [
                    tf.layers.dense(gmp_out, self.config.hidden_dim, name='fc1', kernel_regularizer=regularizer)]
            else:
                fc_list = [
                    tf.layers.dense(self.input_re, self.config.hidden_dim, name='fc1', kernel_regularizer=regularizer,
                                    kernel_initializer=tf.truncated_normal_initializer())]
            fc_list[-1] = tf.contrib.layers.dropout(fc_list[-1], self.keep_prob)
            if self.config.Use_batch_normalization:
                fc_list[-1] = tf.layers.batch_normalization(fc_list[-1], training=self.training, momentum=0.9)
            fc_list[-1] = tf.nn.relu6(fc_list[-1])
            for i in range(self.config.num_hidden_layers - 1):
                fc_list.append(tf.layers.dense(fc_list[-1], self.config.hidden_dim, name='fc1_' + str(i + 1),
                                               kernel_regularizer=regularizer,
                                               kernel_initializer=tf.truncated_normal_initializer()))
                fc_list[-1] = tf.contrib.layers.dropout(fc_list[-1], self.keep_prob)
                if self.config.Use_batch_normalization:
                    fc_list[-1] = tf.layers.batch_normalization(fc_list[-1], training=self.training, momentum=0.9)
                fc_list[-1] = tf.nn.relu6(fc_list[-1])
            # 输出分数

            self.logits = tf.layers.dense(fc_list[-1], 1, name='fc2')
            self.y_pred_cls = self.logits

        with tf.name_scope("optimize"):
            # 损失函数，二范数
            self.loss = tf.losses.mean_squared_error(self.input_y, self.y_pred_cls)
            # 优化器
            self.optim = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            self.acc, self.update_op = tf.contrib.metrics.streaming_pearson_correlation(self.input_y, self.y_pred_cls)
