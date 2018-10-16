import pandas as pd
import os
from collections import defaultdict
from gensim import corpora

base_dir = 'data/101'
train_dir = os.path.join(base_dir, 'trainData_packed.txt')
test_dir = os.path.join(base_dir, 'testData_packed.txt')
val_dir = os.path.join(base_dir, 'validData_packed.txt')
vocab_dir = os.path.join(base_dir, 'My_dic')
forecast_dir = os.path.join(base_dir, 'forecastData_packed.txt')

def load_dic(num):
    if not os.path.exists(vocab_dir):
        f = open("data/" + str(num) + "/original_data/trainData.txt", 'r', encoding='utf8')
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

def read_data():
    Train_Frist_Part = []
    Train_Frist_Part_Tmp = []
    Test_Frist_Part = []
    Test_Frist_Part_Tmp = []

    read_train_filename = 'data/101/original_data/train.xlsx'
    read_test_filename = 'data/101/original_data/testStudent.xlsx'
    write_train_filename = 'data/101/original_data/trainData.txt'
    write_test_filename = 'data/101/original_data/testData.txt'
    write_label_filename = 'data/101/original_data/trainLabel.txt'

    read_train_file = pd.read_excel(read_train_filename)
    read_test_file = pd.read_excel(read_test_filename)
    write_train_tags_file = open(write_train_filename, 'w', encoding='UTF-8')
    write_test_tags_file = open(write_test_filename, 'w', encoding='UTF-8')
    write_label_file = open(write_label_filename, 'w', encoding='UTF-8')

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
        Train_Frist_Part.append(tmp)

    for i in range(len(Test_Frist_Part_Tmp[0])):
        tmp = []
        for j in range(len(Test_Frist_Part_Tmp)):
            tmp.append(Test_Frist_Part_Tmp[j][i])
        Test_Frist_Part.append(tmp)

    dictionary, max_len = load_dic('101')
    return

read_data()