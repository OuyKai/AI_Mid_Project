# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 300  # 词向量维度
    seq_length = 1500  # 序列长度
    num_classes = 101  # 类别数
    num_filters = 200  # 卷积核数目
    kernel_size = 6  # 卷积核尺寸
    vocab_size = -1  # 词汇表大小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 77  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    Three_filter_open = False  # 3种卷积核大小模式
    Use_embedding = True  # 使用word2vec
    Use_batch_normalization = True  #使用BN

    num_hidden_layers = 1  # 隐藏层数量


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        # self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.input_x = tf.placeholder(tf.float32, [None, 6], name='input_x')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # l2_loss = tf.constant(0.0)
        self.cnn()

    def BN(self, input, x):
        wb_mean, wb_var = tf.nn.moments(input, 0)
        scale = tf.Variable(tf.ones([x]))
        offset = tf.Variable(tf.zeros([x]))
        variance_epsilon = 0.001
        return tf.nn.batch_normalization(input, wb_mean, wb_var, offset, scale, variance_epsilon)

    def cnn(self):
        """CNN模型"""
        with tf.name_scope("JB"):
            fc_list = [tf.layers.dense(self.input_x, self.config.hidden_dim, name='fc1')]
            fc_list[-1] = tf.contrib.layers.dropout(fc_list[-1], self.keep_prob)
            if self.config.Use_batch_normalization:
                fc_list[-1] = self.BN(fc_list[-1], self.config.hidden_dim)
            fc_list[-1] = tf.nn.relu6(fc_list[-1])
            for i in range(self.config.num_hidden_layers-1):
                fc_list.append(tf.layers.dense(fc_list[-1], self.config.hidden_dim, name='fc1_'+str(i+1)))
                fc_list[-1] = tf.contrib.layers.dropout(fc_list[-1], self.keep_prob)
                if self.config.Use_batch_normalization:
                    fc_list[-1] = self.BN(fc_list[-1], self.config.hidden_dim)
                fc_list[-1] = tf.nn.relu6(fc_list[-1])
            # 分类器

            self.logits = tf.layers.dense(fc_list[-1], self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name]) * 0.001
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)+0.001*lossL2
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
