#%conda activate tf-gpu
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, Dropout, DepthwiseConv2D, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom, CenterCrop, RandomContrast
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import sklearn.metrics
import tensorflow.keras.backend as kbe
from tensorflow.keras import backend as K
from PIL import Image
import glob
import cv2

class Inference():

    def __init__(self, path : str, train= False, checkpoint_path= 'tmp/hichi', load_model=False):
        self.path = path
        sys.path.append(path)
        if train == True:
            Inference.dataset(self)
            Inference.train(self,checkpoint_filepath=checkpoint_path)
        if load_model == True:
            Inference.load_model()
    
    def dataset(self):
        BATCH_SIZE = 128
        IMG_SIZE = (32, 32)
        training_directory = f"{self.path}/train"
        valid_directory = f"{self.path}/valid"

        self.train_dataset = image_dataset_from_directory(training_directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
        self.validation_dataset = image_dataset_from_directory(valid_directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
    
    def data_augmenter():

        data_augmentation = tf.keras.Sequential()
        data_augmentation.add(RandomContrast(0.2)) 

        return data_augmentation
    
    def block_c(x, power, last_filters):

        batch, input_height, input_width, input_channels = x.shape
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        #x = Dropout(0.1)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), strides=(2, 2),padding='same')(x)
        #x = Dropout(0.1)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        #x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        
        return x
        
    def block_b(x, power, last_filters):
        
        batch, input_height, input_width, input_channels = x.shape
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        #x = Dropout(0.1)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        
        return x

    def block_a(x, power, last_filters):
        
        batch, input_height, input_width, input_channels = x.shape
        x = BatchNormalization(axis=3)(x, training=True)
        input_x = x
        x = Conv2D(power*input_channels, (1, 1), padding='same')(x)
        #x = Dropout(0.1)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        #x = Dropout(0.1)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = ReLU(max_value=6)(x)
        x = Conv2D(last_filters, (1, 1), padding='same')(x)
        #x = Dropout(0.3)(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = Add()([x, input_x])
        
        return x

    def conv_model(input_shape=(32, 32, 3)):

        x_input = Input(shape=input_shape)
        x = Inference.data_augmenter()(x_input)
        x = Conv2D(8, (3, 3), strides=(2, 2),padding='same')(x)
        x = BatchNormalization(axis=3)(x, training=True)
        x = Inference.block_c(x, 1, 8)
        x = Inference.block_c(x, 2, 16)
        x = Inference.block_a(x, 2, 16)
        x = Inference.block_c(x, 2, 24)
        x = Inference.block_a(x, 2, 24)
        x = Inference.block_b(x, 2, 32)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0))(x)
        model = Model(inputs=x_input, outputs=x)
        
        return model
    
    def train(self, model=None, base_learning_rate = 0.01, initial_epochs = 25, checkpoint_filepath = 'tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)', monitor = 'val_accuracy'):
        
        self.model = model
        
        if model == None:
            model = Inference.conv_model()
        
        model.compile(optimizer=Adam(lr=base_learning_rate), loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor=monitor,mode='max', save_best_only=True)
        self.history = model.fit(self.train_dataset, validation_data=self.validation_dataset, epochs=initial_epochs, callbacks=model_checkpoint_callback)

        print('Training process finished!')
        return self.history
    
    def plot_history(self):
        
        acc = [0.] + self.history.history['accuracy']
        val_acc = [0.] + self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
    
    def load_model(self, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):

        model = tf.keras.models.load_model(model_path)
        self.model = model
        
        """elif model_weights_path != None:
            model = Inference.conv_model()
            model.compile(optimizer=Adam(lr=base_learning_rate), loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
            model.load_weights(model_weights_path)
            self.model = model"""

        return model
    
    def predict(self, images_path, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):

        model = Inference.load_model(self, model_path=model_path)
        images = Inference.preprocess_image(images_path)
        predictions = model.predict(images)
        predictions = np.round(predictions, decimals=3)

        """elif model_weights_path != None:
            load_model(self, model_weights_path=model_weights_path)
        images = Inference.preprocess_image(images_path)
        predictions = self.model.predict(images)"""

        return predictions

    def preprocess_image(images_path, type='jpg'):

        images_list = glob.glob(f'{images_path}/*.{type}')
        images = np.zeros([len(images_list), 32, 32, 3])
        
        for i in range(len(images_list)):
            image = cv2.imread(images_list[i])
            if image.shape != (32, 32, 3):
                image = cv2.resize(image, (32, 32))
            image_array = np.asarray(image)
            images[i, :, :, :] = image_array

        return images
    
