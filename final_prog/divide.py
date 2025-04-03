# Data science tools
import numpy as np
import pandas as pd
import os

import shutil

# Image manipulations
from PIL import Image

# Visualizations
import matplotlib.pyplot as plt

DATASET_DIR = '~Desktop/multi-class-food/'
DATASET_PREPROCESS = '~Desktop/dsToProcess/'

os.makedirs(DATASET_PREPROCESS, exist_ok=True)

# Divide dataset
train_dir = DATASET_PREPROCESS + 'train/'
valid_dir = DATASET_PREPROCESS + 'val/'
test_dir = DATASET_PREPROCESS + 'test/'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

categories = []
for folder in os.listdir(DATASET_DIR + 'train'):
    if folder.startswith('.') is False:
        categories.append(folder)

categories.sort()
print(categories)

counter = 0
# For training
for category in categories:
    print(category)
    names = []
    for name_img in os.listdir(DATASET_DIR + 'train/' + category):
        names.append(name_img)

    names.sort()
    #print(names)
    for item in names:
        if counter < 500:
            shutil.copyfile(DATASET_DIR + 'train/' + category + '/' + item, train_dir + category + '/' + item)
            #print(item)
            counter = counter + 1
            #print(counter)

    print(counter)
    counter = 0

# For validation
for category in categories:
    print(category)
    names = []
    for name_img in os.listdir(DATASET_DIR + 'val/' + category):
        names.append(name_img)

    names.sort()
    #print(names)
    for item in names:
        if counter < 250:
            shutil.copyfile(DATASET_DIR + 'val/' + category + '/' + item, valid_dir + category + '/' + item)
            #print(item)
            counter = counter + 1
            #print(counter)

    print(counter)
    counter = 0


# For test
for category in categories:
    print(category)
    names = []
    for name_img in os.listdir(DATASET_DIR + 'test/' + category):
        names.append(name_img)

    names.sort()
    #print(names)
    for item in names:
        if counter < 250:
            shutil.copyfile(DATASET_DIR + 'test/' + category + '/' + item, test_dir + category + '/' + item)
            #print(item)
            counter = counter + 1
            #print(counter)

    print(counter)
    counter = 0

# Example image for testing code
image = Image.open(train_dir + '/bacon/' + 'bacon.1.jpg')

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()
