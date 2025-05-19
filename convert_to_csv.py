# 将data文件夹下的图像（可能需要预处理，如灰度化、调整大小）
# 转换为像素值的CSV文件（如 Data.csv, Abinesh.csv, piolet0016.csv）。
# 图像被展平成一行像素值。
import matplotlib.image as image
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
path = 'Datasets/piolet0016/'
filename = 'piolet0016.csv'
cls = 8

files = os.listdir(path)
dim = (100, 100)
df = pd.DataFrame(columns = [f'pix-{i}' for i in range(1, 1+(dim[0]*dim[1]))]+['class'])
for i in tqdm(range(1, 1+len(files))):
    img =Image.open(path+files[i-1])
    df.loc[i] = list(img.getdata()) + [cls]

df.to_csv(filename,index = False)
print('Task Completed')
