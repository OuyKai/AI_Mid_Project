import os
import random
from collections import defaultdict
from math import sqrt

import pandas as pd
from gensim import corpora

base_dir = 'data/101'
vocab_dir = os.path.join(base_dir, 'My_dic')

def load_dic(num):
    if not os.path.exists(vocab_dir):
        f = open("data/" + str(num) + "/original_data/train_otherData.txt", 'r', encoding='utf8')
        documents = f.readlines()
        # 去掉停用词
        stoplist = set('for a of the and to in'.split())
        # stoplist = set()
        texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in documents]
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

    max_len = 500
    if num == 2:
        max_len = 1463
    return dictionary, max_len


def read_data(num):
    Train_Frist_Part_Tmp = []
    Test_Frist_Part_Tmp = []

    read_train_filename = 'data/' + str(num) + '/original_data/train.xlsx'
    read_test_filename = 'data/' + str(num) + '/original_data/testStudent.xlsx'
    write_train_filename = 'data/' + str(num) + '/original_data/trainData.txt'
    write_test_filename = 'data/' + str(num) + '/original_data/testData.txt'
    write_label_filename = 'data/' + str(num) + '/original_data/trainLabel.txt'
    write_train_other_filename = 'data/' + str(num) + '/original_data/trainOther.txt'
    write_test_other_filename = 'data/' + str(num) + '/original_data/testOther.txt'

    read_train_file = pd.read_excel(read_train_filename)
    read_test_file = pd.read_excel(read_test_filename)
    write_train_tags_file = open(write_train_filename, 'w', encoding='UTF-8')
    write_test_tags_file = open(write_test_filename, 'w', encoding='UTF-8')
    write_label_file = open(write_label_filename, 'w', encoding='UTF-8')
    write_train_other_file = open(write_train_other_filename, 'w', encoding='UTF-8')
    write_test_other_file = open(write_test_other_filename, 'w', encoding='UTF-8')

    Additional_Number_of_Scoring = []
    for data in read_train_file['Additional_Number_of_Scoring']:
        Additional_Number_of_Scoring.append(data)
    Train_Frist_Part_Tmp.append(Additional_Number_of_Scoring)

    Additional_Number_of_Scoring = []
    for data in read_test_file['Additional_Number_of_Scoring']:
        Additional_Number_of_Scoring.append(data)
    Test_Frist_Part_Tmp.append(Additional_Number_of_Scoring)

    Average_Score = []
    for data in read_train_file['Average_Score']:
        Average_Score.append(data)
    Train_Frist_Part_Tmp.append(Average_Score)

    Average_Score = []
    for data in read_test_file['Average_Score']:
        Average_Score.append(data)
    Test_Frist_Part_Tmp.append(Average_Score)

    Review_Total_Negative_Word_Counts = []
    for data in read_train_file['Review_Total_Negative_Word_Counts']:
        Review_Total_Negative_Word_Counts.append(data)
    Train_Frist_Part_Tmp.append(Review_Total_Negative_Word_Counts)

    Review_Total_Negative_Word_Counts = []
    for data in read_test_file['Review_Total_Negative_Word_Counts']:
        Review_Total_Negative_Word_Counts.append(data)
    Test_Frist_Part_Tmp.append(Review_Total_Negative_Word_Counts)

    Total_Number_of_Reviews = []
    for data in read_train_file['Total_Number_of_Reviews']:
        Total_Number_of_Reviews.append(data)
    Train_Frist_Part_Tmp.append(Total_Number_of_Reviews)

    Total_Number_of_Reviews = []
    for data in read_test_file['Total_Number_of_Reviews']:
        Total_Number_of_Reviews.append(data)
    Test_Frist_Part_Tmp.append(Total_Number_of_Reviews)

    Review_Total_Positive_Word_Counts = []
    for data in read_train_file['Review_Total_Positive_Word_Counts']:
        Review_Total_Positive_Word_Counts.append(data)
    Train_Frist_Part_Tmp.append(Review_Total_Positive_Word_Counts)

    Review_Total_Positive_Word_Counts = []
    for data in read_test_file['Review_Total_Positive_Word_Counts']:
        Review_Total_Positive_Word_Counts.append(data)
    Test_Frist_Part_Tmp.append(Review_Total_Positive_Word_Counts)

    Total_Number_of_Reviews_Reviewer_Has_Given = []
    for data in read_train_file['Total_Number_of_Reviews_Reviewer_Has_Given']:
        Total_Number_of_Reviews_Reviewer_Has_Given.append(data)
    Train_Frist_Part_Tmp.append(Total_Number_of_Reviews_Reviewer_Has_Given)

    Total_Number_of_Reviews_Reviewer_Has_Given = []
    for data in read_test_file['Total_Number_of_Reviews_Reviewer_Has_Given']:
        Total_Number_of_Reviews_Reviewer_Has_Given.append(data)
    Test_Frist_Part_Tmp.append(Total_Number_of_Reviews_Reviewer_Has_Given)

    for data in read_train_file['Tags']:
        data = data[1:-1].replace(' ', '').replace(',', ' ').replace('\'', '')
        write_train_tags_file.write(data + '\n')

    for data in read_test_file['Tags']:
        data = data[1:-1].replace(' ', '').replace(',', ' ').replace('\'', '')
        write_test_tags_file.write(data + '\n')

    for data in read_train_file['Reviewer_Score']:
        data = int(data * 10)
        write_label_file.write(str(data) + '\n')

    for i in range(len(Train_Frist_Part_Tmp[0])):
        tmp = []
        for j in range(len(Train_Frist_Part_Tmp)):
            tmp.append(Train_Frist_Part_Tmp[j][i])
        for number in tmp:
            write_train_other_file.write(str(number) + ' ')
        write_train_other_file.write('\n')

    for i in range(len(Test_Frist_Part_Tmp[0])):
        tmp = []
        for j in range(len(Test_Frist_Part_Tmp)):
            tmp.append(Test_Frist_Part_Tmp[j][i])
        write_test_other_file.write(str(0) + '\t')
        for number in tmp:
            write_test_other_file.write(str(number) + ' ')
        write_test_other_file.write('\n')

    # load_dic(num)
    return


def data_pack(num):
    read_train_filename = "data/" + str(num) + "/original_data/trainData.txt"
    read_train_other_filename = "data/" + str(num) + "/original_data/trainOther.txt"
    read_forecast_filename = "data/" + str(num) + "/original_data/testData.txt"
    read_forecast_other_filename = "data/" + str(num) + "/original_data/testOther.txt"
    read_label_filename = "data/" + str(num) + "/original_data/trainLabel.txt"
    if not os.path.exists(read_train_filename):
        read_data(num)
    train = open(read_train_filename, 'r', encoding='utf8')
    other = open(read_train_other_filename, 'r', encoding='UTF-8')
    label = open(read_label_filename, 'r', encoding='utf8')
    forecast = open(read_forecast_filename, 'r', encoding='utf8')
    forecast_other = open(read_forecast_other_filename, 'r', encoding='utf8')
    forecast_data = forecast.readlines()
    forecast_data_other = forecast_other.readlines()
    train_data = train.readlines()
    other_data = other.readlines()
    label_data = label.readlines()

    # shuffle
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_data)
    random.seed(randnum)
    random.shuffle(other_data)
    random.seed(randnum)
    random.shuffle(label_data)

    # write
    new_train = open('data/' + str(num) + '/trainData_packed.txt', 'w', encoding='utf8')
    new_valid = open('data/' + str(num) + '/validData_packed.txt', 'w', encoding='utf8')
    new_test = open('data/' + str(num) + '/testData_packed.txt', 'w', encoding='utf8')
    new_forecast = open('data/' + str(num) + '/forecastData_packed.txt', 'w', encoding='utf8')
    new_other_train = open('data/' + str(num) + '/train_otherData_packed.txt', 'w', encoding='utf8')
    new_other_valid = open('data/' + str(num) + '/valid_otherData_packed.txt', 'w', encoding='utf8')
    new_other_test = open('data/' + str(num) + '/test_otherData_packed.txt', 'w', encoding='utf8')
    new_other_forecast = open('data/' + str(num) + '/forecast_otherData_packed.txt', 'w', encoding='utf8')

    for i in range(int(len(train_data) * 3 / 5)):
        tmp = str(label_data[i].strip()) + '\t' + train_data[i]
        new_train.write(tmp)
        # new_other_train.write(other_data[i])
        tmp = str(label_data[i].strip()) + '\t' + other_data[i]
        new_other_train.write(tmp)
    for i in range(int(len(train_data) * 3 / 5), int(len(train_data) * 4 / 5)):
        tmp = str(label_data[i].strip()) + '\t' + train_data[i]
        new_valid.write(tmp)
        # new_other_valid.write(other_data[i])
        tmp = str(label_data[i].strip()) + '\t' + other_data[i]
        new_other_valid.write(tmp)
    for i in range(int(len(train_data) * 4 / 5), len(train_data)):
        tmp = str(label_data[i].strip()) + '\t' + train_data[i]
        new_test.write(tmp)
        # new_other_test.write(other_data[i])
        tmp = str(label_data[i].strip()) + '\t' + other_data[i]
        new_other_test.write(tmp)
    for i in range(len(forecast_data)):
        tmp = '0' + '\t' + forecast_data[i]
        new_forecast.write(tmp)
        tmp = forecast_data_other[i]
        new_other_forecast.write(tmp)

    train.close()
    other.close()
    label.close()
    forecast.close()
    new_test.close()
    new_train.close()
    new_valid.close()
    new_forecast.close()
    new_other_test.close()
    new_other_train.close()
    new_other_valid.close()
    new_other_forecast.close()


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(X, Y):
    A = []
    B = []
    for i in X:
        A.append(i / 10)
    for i in Y:
        B.append(i / 10)
    n = len(A)
    # 求和
    sum1 = sum(A)
    sum2 = sum(B)
    # 求乘积之和
    sumofxy = multipl(A, B)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in A])
    sumofy2 = sum([pow(j, 2) for j in B])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den
