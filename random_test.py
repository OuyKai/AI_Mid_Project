import os

max_num = 4  # 总运行次数

for i in range(max_num):
    print('The ',i+1,' Round:')
    os.system('python -W ignore run_cnn.py')
    # time.sleep(60)
