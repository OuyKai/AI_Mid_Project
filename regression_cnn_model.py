# coding: utf-8

import pickle

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 77  # 词向量维度
    seq_length = 1500  # 序列长度
    num_classes = 101  # 类别数
    num_filters = 777  # 卷积核数目
    kernel_size = 7  # 卷积核尺寸
    vocab_size = -1  # 词汇表达小

    hidden_dim = 77  # 全连接层神经元

    dropout_keep_prob = 0.7  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 77  # 每批训练大小
    num_epochs = 6  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    Three_filter_open = False  # 3种卷积核大小模式
    Use_embedding = True  # 使用glove
    Use_batch_normalization = True  #使用BN


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.input_re = tf.placeholder(tf.int32, [None, 6], name='input_re')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        l2_loss = tf.constant(0.0)
        self.cnn()

    def BN(self, input, x):
        wb_mean, wb_var = tf.nn.moments(input, 0)
        scale = tf.Variable(tf.ones([x]))
        offset = tf.Variable(tf.zeros([x]))
        variance_epsilon = 0.001
        return tf.nn.batch_normalization(input, wb_mean, wb_var, offset, scale, variance_epsilon)

    def cnn(self):
        """CNN模型"""

        # 词向量加载
        with open("glove_word_vector.pkl", 'rb') as f:
            embedding_weights = pickle.load(f)

        # 词向量映射
        with tf.device('/cpu:0'):
            if self.config.Use_embedding:
                embedding = tf.Variable(embedding_weights, trainable=False, name='embedding_weights', dtype='float32')
            else:
                embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)


        with tf.name_scope("cnn"):
            # CNN layer
            conv_0 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv_0')
            if self.config.Three_filter_open:
                conv_1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size-1, name='conv_1')
                conv_2 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size+1, name='conv_2')

            # global max pooling layer

            gmp_0 = tf.reduce_max(conv_0, reduction_indices=[1], name='gmp_0')
            gmp_0 = tf.concat([gmp_0, self.input_re], 1, name='combine')
            if self.config.Three_filter_open:
                gmp_1 = tf.reduce_max(conv_1, reduction_indices=[1], name='gmp_1')
                gmp_2 = tf.reduce_max(conv_2, reduction_indices=[1], name='gmp_2')

                gmp_all = tf.concat([gmp_0, gmp_1, gmp_2,self.input_re], 1, name='combine')


        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            if self.config.Three_filter_open:
                gmp_all = self.BN(gmp_all, self.config.num_filters * 3)
                fc = tf.layers.dense(gmp_all, self.config.hidden_dim, name='fc1')
            else:
                gmp_0 = self.BN(gmp_0, self.config.num_filters)
                fc = tf.layers.dense(gmp_0, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            if self.config.Use_batch_normalization:
                fc = self.BN(fc, self.config.hidden_dim)
            fc = tf.nn.relu6(fc)
            if self.config.Use_batch_normalization:
                fc = self.BN(fc, self.config.hidden_dim)
            fc_1 = tf.layers.dense(fc, self.config.hidden_dim, name='fc1_1')
            if self.config.Use_batch_normalization:
                fc_1 = self.BN(fc_1, self.config.hidden_dim)
            fc_1 = tf.nn.relu6(fc_1)
            # 分类器

            self.logits = tf.layers.dense(fc_1, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
