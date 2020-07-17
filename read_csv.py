# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:00:06 2020

@author: gaoyilei
"""


import pandas as pd
import numpy as np

from skimage import io
import random

import csv

#from sklearn.model_select import train_test_split

data = pd.read_csv('./new_breast.csv')

data_num = len(data)

train_num = int(data_num*0.5)
val_num = int(data_num*0.1)
test_num = data_num - train_num - val_num

train_data = data[:train_num]
val_data = data[train_num:(train_num+val_num)]
test_data = data[(train_num+val_num):]

train_data.to_csv('./dataset/train.csv')
val_data.to_csv('./dataset/val.csv')
test_data.to_csv('./dataset/test.csv')



# path_list = df_breast['Name']
# label_list = df_breast['Label']


# index_x = [i for i in range(len(path_list))]
# random.shuffle(index_x)

           
# f = open('new_breast.csv', 'w', encoding='utf-8', newline="")

# csv_writer = csv.writer(f)

# csv_writer.writerow(['Name', 'Label'])
# # csv_writer.writerow(['q', '1'])

# for i in range(len(path_list)):
#     new_path = path_list[index_x[i]]
#     new_label = label_list[index_x[i]]
#     csv_writer.writerow([new_path, new_label])

# f.close()
           