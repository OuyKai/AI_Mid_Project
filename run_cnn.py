#!/usr/bin/python
# -*- coding: utf-8 -*-

# from __future__ import print_function

import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from datetime import timedelta

import numpy as np
import tensorflow as tf
from gensim import corpora
from gensim.models import word2vec
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.load_helper import read_category, batch_iter, process_file
from data_pack import data_pack

num_classes = 5  # Attention!!!!!!!!!!!!!!!

base_dir = 'data/' + str(num_classes)
train_dir = os.path.join(base_dir, 'trainData_packed.txt')
test_dir = os.path.join(base_dir, 'testData_packed.txt')
val_dir = os.path.join(base_dir, 'validData_packed.txt')
vocab_dir = os.path.join(base_dir, 'My_dic')
forecast_dir = os.path.join(base_dir, 'forecastData_packed.txt')

result_dir = 'output'

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def forecast():
    print("Loading forecast data...")
    start_time = time.time()
    x_test, y_test = process_file(forecast_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Forecasting...')

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 输出
    filename_prefix = '16337158_'
    filename_suffix = '.txt'
    filenum = 0
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    while os.path.exists(result_dir + '/' + filename_prefix + str(filenum) + filename_suffix):
        filenum += 1
    f = open(result_dir + '/' + filename_prefix + str(filenum) + filename_suffix, 'w', encoding='utf8')
    hh = list(y_pred_cls)
    for i in range(len(hh)):
        f.write(str(hh[i]) + '\n')
    f.close()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return msg.format(loss_test, acc_test), loss_test, acc_test


def log_and_clean(test_result, para, loss_test, acc_test):
    filename_prefix = 'log/para_'
    filename_prefix_1 = 'log/result_'
    filename_suffix = '.txt'
    filenum = 0
    if not os.path.exists('log'):
        os.makedirs('log')
    while os.path.exists(filename_prefix + str(filenum) + filename_suffix):
        filenum += 1
    dic = {}
    dic['embedding_dim'] = para.embedding_dim
    dic['seq_length'] = para.seq_length
    dic['num_classes'] = para.num_classes
    dic['num_filters'] = para.num_filters
    dic['kernel_size'] = para.kernel_size
    dic['vocab_size'] = para.vocab_size
    dic['hidden_dim'] = para.hidden_dim
    dic['dropout_keep_prob'] = para.dropout_keep_prob
    dic['learning_rate'] = para.learning_rate
    dic['batch_size'] = para.batch_size
    dic['num_epochs'] = para.num_epochs
    f1 = open(filename_prefix + str(filenum) + filename_suffix, 'w')
    f2 = open(filename_prefix_1 + str(filenum) + filename_suffix, 'w')
    f1.write(json.dumps(dic))
    f2.write(test_result)

    # 记录最好结果
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists('log/best_result.txt'):
        f = open('log/best_result.txt', 'w', encoding='utf8')
        dd = {'best_loss': loss_test, 'best_accuracy': acc_test, 'file_num': filenum}
        f.write(json.dumps(dd))
    else:
        f = open('log/best_result.txt', 'r', encoding='utf8')
        dd = json.loads(f.read())
        if acc_test > dd['best_accuracy']:
            dd['best_loss'] = loss_test
            dd['best_accuracy'] = acc_test
            dd['file_num'] = filenum
            f.close()
            f = open('log/best_result.txt', 'w', encoding='utf8')
            f.write(json.dumps(dd))
    f.close()
    f1.close()
    f2.close()
    copy_to_save('checkpoints')


def copy_to_save(path):
    flag = True
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            if flag:
                filename_prefix = 'log/'
                filenum = 0
                while os.path.exists(filename_prefix + str(filenum) + '/'):
                    filenum += 1
                os.mkdir(filename_prefix + str(filenum) + '/')
            hh = path_file[len(path):]
            shutil.copyfile(path_file, filename_prefix + str(filenum) + hh)
            flag = False
        else:
            copy_to_save(path_file)


def load_dic(num):
    if not os.path.exists(vocab_dir):
        f = open("data/" + str(num) + "/original_data/trainData.txt", 'r', encoding='utf8')
        documents = f.readlines()
        o = open("data/" + str(num) + "/original_data/trainData_new.txt", 'w', encoding='utf8')

        # 去掉停用词
        stoplist = set('for a of the and to in'.split())
        # stoplist = set()
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in documents]
        for line in texts:
            for word in line:
                o.write(word + " ")
            o.write('\n')

        # 去掉只出现一次的单词
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1]
                 for text in texts]
        max_len = 0
        for i in texts:
            if len(i) > max_len:
                max_len = len(i)
        dictionary = corpora.Dictionary(texts)
        dictionary.save(vocab_dir)
        f.close()
    else:
        dictionary = corpora.Dictionary.load(vocab_dir)

    max_len = 1500
    if num == 2:
        max_len = 1463
    return dictionary, max_len


#制作词向量矩阵
def build_word_array(word_to_id, model,item):
    data={}
    vector_array=[]
    word_to_id_copy = word_to_id.copy()
    if item==1:
        with open("word_vector.pkl", 'wb') as o:
            for i in word_to_id.keys():
                vector_array.append(list(model[i]))

    else:
        with open("word_vector.pkl", 'wb') as o, \
                open("vectors" + str(num_classes) + ".txt", 'r', encoding='utf8') as f:
            # for i in word_to_id.keys():
            # vector_array.append(list(model[i]))
            for line in f.readlines():
                num = 0
                line_vector = []
                line_key = ''
                for word in line.split():
                    if num == 0:
                        line_key = word
                        num += 1
                    else:
                        line_vector.append(float(word.strip()))
                data[line_key] = line_vector
            max_len = 0
            del_num = 0
            for line in word_to_id.keys():
                # print(line)
                word_to_id_copy[line] -= del_num
                if line in data.keys():
                    vector_array.append(data[line])
                    if len(line) > max_len:
                        max_len = len(line)
                else:
                    word_to_id_copy.pop(line)
                    del_num += 1
        pickle.dump(vector_array, o)
        max_len = 1453
        return word_to_id_copy,max_len

def build_vector(fileName):
    if os.path.exists("data/" + str(num_classes) + "/" + str(num_classes) + "model.model"):
        model = word2vec.Word2Vec.load("data/" + str(num_classes) + "/" + str(num_classes) + "model.model")
    else:
        sentences = word2vec.Text8Corpus(fileName)
        model = word2vec.Word2Vec(sentences, size=200, min_count=0)
        model.save("data/" + str(num_classes) + "/" + str(num_classes) + "model.model")
    return model


if __name__ == '__main__':
    print('Generating data...')
    data_pack(num_classes)
    print('Configuring CNN model...')
    config = TCNNConfig()
    config.num_classes = num_classes
    print('Building dictionary...')
    dictionary, max_len = load_dic(num_classes)
    # dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
    word_to_id = dictionary.token2id
    id_to_word = dictionary.id2token
    print('Performing word2vec...')
    if config.Use_embedding:
        config.choose_wordVector=0  #0是glove,1是word2vec
        model = build_vector("data/" + str(num_classes) + "/original_data/trainData_new.txt")
        word_to_id, max_len = build_word_array(word_to_id, model,config.choose_wordVector)

    words = list(word_to_id.keys())
    categories, cat_to_id = read_category(num_classes)
    config.vocab_size = len(words)
    # config.seq_length = max_len
    model = TextCNN(config)
    with open("glove_word_vector.pkl", 'rb') as f:
        embedding_weights = pickle.load(f)
    train()
    forecast()
    log, loss_test, acc_test = test()
    log_and_clean(log, config, loss_test, acc_test)
    print('Completed!')
