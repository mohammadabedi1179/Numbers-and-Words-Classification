#python 3
import sys
import tensorflow as tf
import tensorflow.keras.layers as tkl
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
#from keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom, CenterCrop, RandomContrast
from tensorflow.keras.metrics import AUC
import numpy as np
import sklearn.metrics
import tensorflow.keras.backend as kbe
#from keras import backend as K

class Inference():

    def __init__(self, path):
        self.path = path
        sys.path.append(path)
    
    def dataset(self):
        BATCH_SIZE = 128
        IMG_SIZE = (32, 32)
        training_directory = f"{self.path}/train"
        valid_directory = f"{self.path}/valid"

        train_dataset = image_dataset_from_directory(training_directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
        validation_dataset = image_dataset_from_directory(valid_directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
    
    def data_augmenter():
        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(RandomContrast(0.2)) 
        #data_augmentation.add(CenterCrop(20, 20))
        #data_augmentation.add(RandomZoom((-0.4, 0.2)))
    
    
        return data_augmentation
    
    