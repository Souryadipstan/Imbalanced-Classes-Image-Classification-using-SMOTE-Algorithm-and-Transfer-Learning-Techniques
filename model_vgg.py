import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from keras.layers import Flatten,Dense, Input, AveragePooling2D, GlobalAveragePooling2D,Dropout,Add,ZeroPadding2D, MaxPooling2D,Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.metrics import FBetaScore
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

df_train = pd.read_csv("image-classification-2/train_label.csv",dtype = 'str')
# df_train=df_train.sort_values('label')

img_width = 224
img_height = 224

IMAGE_DIR = 'image-classification-2/train/'
X_train = []

if os.path.isfile('X_train_224.npy'):

    X_train = np.load('X_train_224.npy') # load

else:

    for img in df_train['file_name']:
        loaded_img = load_img(os.path.join(IMAGE_DIR, img), target_size=(img_width, img_height))
        img_arr = img_to_array(loaded_img)
        X_train.append(img_arr)

    np.save('X_train_224.npy', X_train) # save
        
print(np.array(X_train).shape)  
y_train = df_train.drop('file_name', axis=1, inplace=False)
print(y_train.head())
y_train = np.array(y_train.values)
X_train = np.array(X_train)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train.reshape((-1, img_width * img_height * 3)), y_train)

X_train = X_train.reshape(-1, img_width, img_height, 3)

for i in range(len(df_train)):
    df_train.loc[i,'file_name'] = 'image-classification-2/train/'+df_train.loc[i,'file_name']


strat_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=2)

for tr_index, val_index in strat_split.split(df_train, df_train.loc[:,'label']):
    strat_train_df = df_train.loc[tr_index]
    strat_valid_df = df_train.loc[val_index]

y_train = to_categorical(y_train, 11)

y_train = np.asarray(y_train).astype(np.float32)

raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df_train, 
        x_col="file_name", 
        y_col='label', 
        class_mode="categorical", 
        batch_size=64, 
        shuffle=True, 
        target_size=(img_width, img_height))

batch = raw_train_generator.next()
data_sample = batch[0]

valid_data_IDG = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
valid_data_IDG.fit(data_sample)

train_data_IDG = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)

X_train_1, X_test, y_train_1, y_test = train_test_split(X_train,y_train,test_size = 0.2)

training_data = train_data_IDG.flow(X_train_1,y_train_1, batch_size=64, shuffle=False)

validation_data = valid_data_IDG.flow(X_test,y_test, shuffle=False)

model = load_model("Saved_models/model.hdf5")

df_test = pd.read_csv('image-classification-2/sample_submission.csv',dtype = 'str')

for i in range(len(df_test)):
    df_test.loc[i,'file_name'] = 'image-classification-2/test/'+df_test.loc[i,'file_name']

test_generator = valid_data_IDG.flow_from_dataframe(
        df_test,
        target_size=(img_width,img_height),
        batch_size=1,
        class_mode='categorical',y_col='label', x_col='file_name',
        shuffle=False)

pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "label":predicted_class_indices})

for i in range(len(results)):
    
    results.loc[i,'file_name'] = results.loc[i,'file_name'].replace('image-classification-2/test/','')

results.to_csv("result.csv",index=False)
