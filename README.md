# XAI-
# first insure that the cache is empty to avoid error appearance by running those commands 

* !pip uninstall scikit-learn -y
* !pip cache purge
* !pip install scikit-learn --upgrade

# secong import those libraries :

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.image import imread

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, concatenate,
    BatchNormalization, Dropout, MaxPooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import keras.utils as image

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Oversampling
from imblearn.over_sampling import SMOTE

# Misc
import glob
import PIL
import random
random.seed(100)

# Confirm versions
print("TensorFlow version:", tf.__version__)
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# for plots at the last two cells run those 

import os
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
# PRESENTATION LINK 
https://www.canva.com/design/DAGmfvh5TEk/s3JOse9oRilvLCwx_f9JQQ/edit
