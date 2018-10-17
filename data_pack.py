import random

def data_pack(num):
    read_train_filename = "data/"+str(num)+"/original_data/trainData.txt"
    read_label_filename = "data/"+str(num)+"/original_data/trainLabel.txt"
    train = open(read_train_filename, 'r', encoding='utf8')
    label = open(read_label_filename, 'r', encoding='utf8')
    train_data = train.readlines()
    label_data = label.readlines()
    # shuffle
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_data)
    random.seed(randnum)
    random.shuffle(label_data)
    # write
    new_train = open('data/'+str(num)+'/trainData_packed.txt', 'w', encoding='utf8')
    new_valid = open('data/'+str(num)+'/validData_packed.txt','w',encoding='utf8')
    new_test = open('data/'+str(num)+'/testData_packed.txt', 'w', encoding='utf8')
    for i in range(int(len(train_data)*3/5)):
        tmp = label_data[i][0] + '\t' + train_data[i]
        new_train.write(tmp)
    for i in range(int(len(train_data)*3/5),int(len(train_data)*4/5)):
        tmp = label_data[i][0] + '\t' + train_data[i]
        new_valid.write(tmp)
    for i in range(int(len(train_data)*4/5),len(train_data)):
        tmp = label_data[i][0] + '\t' + train_data[i]
        new_test.write(tmp)
    train.close()
    label.close()
    new_test.close()
    new_train.close()
    new_valid.close()
