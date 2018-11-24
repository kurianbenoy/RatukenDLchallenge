import pandas as pd
import shutil
import os

source = '/home/kurianbwnoy/Data/Ratuken_data/training-data/train-images/'
dest = '/home/kurianbwnoy/Data/Ratuken_data/sorted-data/'

df = pd.read_csv('/home/kurianbwnoy/Data/Ratuken_data/training-data/train.csv')
print(df.index)

df_length = df.shape[0]

for index, row in df.iterrows():
    print('coping:',source+row['image'], dest+str(row['cat']),'\t\t',(index/df_length)*100, '%')
    shutil.copy(source+row['image'], dest+str(row['cat']))


    # for i in range(43):
    #     os.mkdir(dest+str(i))