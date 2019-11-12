
# coding: utf-8

# In[1]:


print("starting")


# In[2]:
#RSNA Intracranial Hemorrhage Detection using EfficientNet ensemble B2-B5. This is a multiclass classification problem.
#We perform one full pass through the data, then multiple epochs on a subset of the data and averaging the predictions.

#Data can be downloaded at: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data



#Perform imports
import numpy as np
import pandas as pd
import pydicom
import os
import collections
import sys
import glob
import random
import cv2
import tensorflow as tf
import multiprocessing

from math import ceil, floor
from copy import deepcopy
from tqdm import tqdm
from imgaug import augmenters as iaa

import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import optimizers


# In[2]:




# In[4]:


# Import Custom Modules
import efficientnet.keras as efn 
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# In[5]:


# Set seed
SEED = 12345
np.random.seed(SEED)
tf.set_random_seed(SEED)

# Constants
TEST_SIZE = 0.06
HEIGHT = 256
WIDTH = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16

# Folders: These lead to location of images folders
DATA_DIR = '/home/sergei/rsna/'
TEST_IMAGES_DIR = DATA_DIR + 'ann/stage_1_test_images/'
TRAIN_IMAGES_DIR = DATA_DIR + 'stage_1_train_images/'


# In[7]:

#DICOM windowing and data generators
def _get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def _get_windowing(data):
    dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def _window_image(img, window_center, window_width, slope, intercept):
    img = (img * slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def _read(path, desired_size=(224, 224)):
    dcm = pydicom.dcmread(path)
    window_params = _get_windowing(dcm) # (center, width, slope, intercept)

    try:
        # dcm.pixel_array might be corrupt
        img = _window_image(dcm.pixel_array, *window_params)
    except:
        img = np.zeros(desired_size)

    img = _normalize(img)

    if desired_size != (512, 512):
        # resize image
        img = cv2.resize(img, desired_size, interpolation = cv2.INTER_LINEAR)
    return img[:,:,np.newaxis]


# In[8]:


# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.25, aug)
augmentation = iaa.Sequential([  
                                iaa.Fliplr(0.25),
                                sometimes(iaa.Crop(px=(0, 25), keep_size = True, sample_independently = False))   
                            ], random_order = True)       
        
# Generators
class TrainDataGenerator(keras.utils.Sequence):

    def __init__(self, dataset, labels, batch_size=16, img_size=(512, 512), img_dir = TRAIN_IMAGES_DIR, augment = False, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y

    def augmentor(self, image):
        augment_img = augmentation        
        image_aug = augment_img.augment_image(image)
        return image_aug

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size, 3))
        Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = _read(self.img_dir+ID+".dcm", self.img_size)
            if self.augment:
                X[i,] = self.augmentor(image)
            else:
                X[i,] = image            
            Y[i,] = self.labels.iloc[index].values        
        return X, Y
    
class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, ids, labels, batch_size = 5, img_size = (512, 512), img_dir = TEST_IMAGES_DIR, *args, **kwargs):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.ids[k] for k in indices]
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size, 3))
        for i, ID in enumerate(list_IDs_temp):
            image = _read(self.img_dir+ID+".dcm", self.img_size)
            X[i,] = image            
        return X


# In[13]:
#Read train and test sets

def read_testset(filename = DATA_DIR + "/ann/stage_1_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

def read_trainset(filename = DATA_DIR + "/ann/stage_1_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468,  312469,  312470,  312471,  312472,  312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]
    df = df.drop(index = duplicates_to_remove)
    df = df.reset_index(drop = True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

# Read Train and Test Datasets
test_df = read_testset()
train_df = read_trainset()


# In[14]:


# Oversampling
epidural_df = train_df[train_df.Label['epidural'] == 1]
train_oversample_df = pd.concat([train_df, epidural_df])
train_df = train_oversample_df

# Summary
print('Train Shape: {}'.format(train_df.shape))
print('Test Shape: {}'.format(test_df.shape))


# In[3]:


def predictions(test_df, model):    
    test_preds = model.predict_generator(TestDataGenerator(test_df.iloc[range(test_df.shape[0])].index, None, 5, (WIDTH, HEIGHT), TEST_IMAGES_DIR), verbose=1)
    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'min', 
                            period = 1)




















# Create Model B2
def create_model():
    K.clear_session()
    
    base_model =  efn.EfficientNetB2(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (HEIGHT, WIDTH, 3))
    x = base_model.output
    x = Dropout(0.125)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


# In[16]:


# Submission Placeholder
submission_predictions_b2 = []



# Multi Label Stratified Split stuff
msss = MultilabelStratifiedShuffleSplit(n_splits = 20, test_size = TEST_SIZE, random_state = SEED)
X = train_df.index
Y = train_df.Label.values

# Get train and test index
msss_splits = next(msss.split(X, Y))
train_idx = msss_splits[0]
valid_idx = msss_splits[1]


# In[17]:


# Loop through Folds of Multi Label Stratified Split
#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 
#    # Get train and test index
#    train_idx = msss_splits[0]
#    valid_idx = msss_splits[1]
for epoch in range(0, 12):
    print('=========== EPOCH {}'.format(epoch))

    # Shuffle Train data
    np.random.shuffle(train_idx)
    print(train_idx[:5])    
    print(valid_idx[:5])

    # Create Data Generators for Train and Valid
    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 
                                                train_df.iloc[train_idx], 
                                                TRAIN_BATCH_SIZE, 
                                                (WIDTH, HEIGHT),
                                                augment = True)
    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 
                                            train_df.iloc[valid_idx], 
                                            VALID_BATCH_SIZE, 
                                            (WIDTH, HEIGHT),
                                            augment = False)

    # Create Model
    model = create_model()
    
    # Head Training Model
    if epoch < 1:
        for base_layer in model.layers[:-5]:
            base_layer.trainable = False
        TRAIN_STEPS = int(len(data_generator_train) /  1)
        LR = 0.0004
    # Full Training Model
    else:
        for base_layer in model.layers[:-1]:
            base_layer.trainable = True
        TRAIN_STEPS = int(len(data_generator_train) / 6)
        LR = 0.0001

    if epoch != 0:
        # Load Model Weights
        print("loading weights")
        model.load_weights('/home/sergei/rsna/eff_b2/model_b2_8_sub' + str(epoch - 1) + '.h5')    
    #AD = Adam(lr = LR)
    model.compile(optimizer = Adam(learning_rate = LR), 
                  loss = 'binary_crossentropy',
                  metrics = ['acc', tf.keras.metrics.AUC()])
    
    # Train Model
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = TRAIN_STEPS,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull('/home/sergei/rsna/eff_b2/model_b2_8_sub' + str(epoch) + '.h5')],
                        verbose = 1)
    
    # Starting with epoch 4 we create predictions for the test set on each epoch
    if epoch > 3:
        preds = predictions(test_df, model)
        submission_predictions_b2.append(preds)
    
    import pickle

    output = open('/home/sergei/rsna/eff_b2/hist/b2e' + str(epoch) + '.pkl', 'wb')
    pickle.dump(submission_predictions_b2, output)
    output.close()  
    print('Saved' + 'b2e' + str(epoch) + '.pkl')
    print("Epoch" + str(epoch) + " has been a great success!!!")

        
        
        
        
        
        
        
        
        
    
        
        
        


# In[2]:


#Repeat for B3


TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16

def create_model_b3():
    K.clear_session()
    
    base_model =  efn.EfficientNetB3(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (HEIGHT, WIDTH, 3))
    x = base_model.output
    x = Dropout(0.125)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


# In[16]:


# Submission Placeholder
submission_predictions_b3 = []











# Multi Label Stratified Split stuff...
msss = MultilabelStratifiedShuffleSplit(n_splits = 20, test_size = TEST_SIZE, random_state = SEED)
X = train_df.index
Y = train_df.Label.values

# Get train and test index
msss_splits = next(msss.split(X, Y))
train_idx = msss_splits[0]
valid_idx = msss_splits[1]


# In[17]:


# Loop through Folds of Multi Label Stratified Split
#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 
#    # Get train and test index
#    train_idx = msss_splits[0]
#    valid_idx = msss_splits[1]
for epoch in range(0, 12):
    print('=========== EPOCH {}'.format(epoch))

    # Shuffle Train data
    np.random.shuffle(train_idx)
    print(train_idx[:5])    
    print(valid_idx[:5])

    # Create Data Generators for Train and Valid
    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 
                                                train_df.iloc[train_idx], 
                                                TRAIN_BATCH_SIZE, 
                                                (WIDTH, HEIGHT),
                                                augment = True)
    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 
                                            train_df.iloc[valid_idx], 
                                            VALID_BATCH_SIZE, 
                                            (WIDTH, HEIGHT),
                                            augment = False)

    # Create Model
    model = create_model_b3()
    
    # Head Training Model
    if epoch < 1:
        for base_layer in model.layers[:-5]:
            base_layer.trainable = False
        TRAIN_STEPS = int(len(data_generator_train) /  1)
        LR = 0.0004
    # Full Training Model
    else:
        for base_layer in model.layers[:-1]:
            base_layer.trainable = True
        TRAIN_STEPS = int(len(data_generator_train) / 6)
        LR = 0.0001

    if epoch != 0:
        # Load Model Weights
        print("loading weights")
        model.load_weights('/home/sergei/rsna/eff_b2/model_b3_8_sub' + str(epoch - 1) + '.h5')    
    #AD = Adam(lr = LR)
    model.compile(optimizer = Adam(learning_rate = LR), 
                  loss = 'binary_crossentropy',
                  metrics = ['acc', tf.keras.metrics.AUC()])
    
    # Train Model
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = TRAIN_STEPS,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull('/home/sergei/rsna/eff_b2/model_b3_8_sub' + str(epoch) + '.h5')],
                        verbose = 1)
    
    # Starting with epoch 4 we create predictions for the test set on each epoch
    if epoch > 3:
        preds = predictions(test_df, model)
        submission_predictions_b3.append(preds)
    
    import pickle

    output = open('/home/sergei/rsna/eff_b2/hist/b3e' + str(epoch) + '.pkl', 'wb')
    pickle.dump(submission_predictions_b3, output)
    output.close()  
    print('Saving ' + 'b3e' + str(epoch) + '.pkl')
    print("Epoch" + str(epoch) + " has been a great success!!!")

        


# In[3]:


# Repeat for B4

TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 4

def create_model_b4():
    K.clear_session()
    
    base_model =  efn.EfficientNetB4(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (HEIGHT, WIDTH, 3))
    x = base_model.output
    x = Dropout(0.125)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


# In[16]:


# Submission Placeholder
submission_predictions_b4 = []

# Multi Label Stratified Split stuff...
msss = MultilabelStratifiedShuffleSplit(n_splits = 20, test_size = TEST_SIZE, random_state = SEED)
X = train_df.index
Y = train_df.Label.values

# Get train and test index
msss_splits = next(msss.split(X, Y))
train_idx = msss_splits[0]
valid_idx = msss_splits[1]


# In[17]:


# Loop through Folds of Multi Label Stratified Split
#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 
#    # Get train and test index
#    train_idx = msss_splits[0]
#    valid_idx = msss_splits[1]
for epoch in range(0, 12):
    print('=========== EPOCH {}'.format(epoch))

    # Shuffle Train data
    np.random.shuffle(train_idx)
    print(train_idx[:5])    
    print(valid_idx[:5])

    # Create Data Generators for Train and Valid
    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 
                                                train_df.iloc[train_idx], 
                                                TRAIN_BATCH_SIZE, 
                                                (WIDTH, HEIGHT),
                                                augment = True)
    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 
                                            train_df.iloc[valid_idx], 
                                            VALID_BATCH_SIZE, 
                                            (WIDTH, HEIGHT),
                                            augment = False)

    # Create Model
    model = create_model_b4()
    
    # Head Training Model
    if epoch < 1:
        for base_layer in model.layers[:-5]:
            base_layer.trainable = False
        TRAIN_STEPS = int(len(data_generator_train) /  1)
        LR = 0.0004
    # Full Training Model
    else:
        for base_layer in model.layers[:-1]:
            base_layer.trainable = True
        TRAIN_STEPS = int(len(data_generator_train) / 6)
        LR = 0.0001

    if epoch != 0:
        # Load Model Weights
        print("loading weights")
        model.load_weights('/home/sergei/rsna/eff_b2/model_b4_8_sub' + str(epoch - 1) + '.h5')    
    #AD = Adam(lr = LR)
    model.compile(optimizer = Adam(learning_rate = LR), 
                  loss = 'binary_crossentropy',
                  metrics = ['acc', tf.keras.metrics.AUC()])
    
    # Train Model
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = TRAIN_STEPS,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull('/home/sergei/rsna/eff_b2/model_b4_8_sub' + str(epoch) + '.h5')],
                        verbose = 1)
    
    # Starting with epoch 4 we create predictions for the test set on each epoch
    if epoch > 3:
        preds = predictions(test_df, model)
        submission_predictions_b4.append(preds)
      
    import pickle

    output = open('/home/sergei/rsna/eff_b2/hist/b4e' + str(epoch) + '.pkl', 'wb')
    pickle.dump(submission_predictions_b4, output)
    output.close()  
    print('Saved' + 'b4e' + str(epoch) + '.pkl')
    print("Epoch" + str(epoch) + " has been a great success!!!")
        

# In[ ]:


# In[ ]:


final = submission_predictions_b2 + submission_predictions_b3 + submission_predictions_b4


# In[ ]:


test_df.iloc[:, :] = np.average(submission_predictions, axis = 0, weights = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128])
test_df = test_df.stack().reset_index()
test_df.insert(loc = 0, column = 'ID', value = test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('final_submission.csv', index = False)

