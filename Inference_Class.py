#python3
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, ReLU, BatchNormalization, Dropout, DepthwiseConv2D, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import sklearn.metrics
import glob
import cv2
import pandas as pd
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
        self.calass_names = ['alphabet', 'number']
        self.batch_size = 128
        IMG_SIZE = (32, 32)
        training_directory = f"{self.path}/train"
        valid_directory = f"{self.path}/valid"

        self.train_dataset = image_dataset_from_directory(training_directory,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
                                                    image_size=IMG_SIZE,
                                                    seed=42)
        self.validation_dataset = image_dataset_from_directory(valid_directory,
                                                    shuffle=True,
                                                    batch_size=self.batch_size,
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

        return model
    
    def predict(self, images_path, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):

        model = Inference.load_model(self, model_path=model_path)
        images = Inference.preprocess_image(images_path)
        predictions = model.predict(images)
        predictions = np.round(predictions, decimals=3)

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
    
    def eer(numpy_labels, numpy_labels_pred, positive_label=1):

        fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy_labels, numpy_labels_pred, pos_label=positive_label)
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        eer = eer_1/2 + eer_2/2

        return eer
    
    def compute_eer(self, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):
        Inference.dataset(self)
        Inference.load_model(self, model_path=model_path)

        train_numpy_labels, train_numpy_labels_pred = Inference.create_label_and_pred(self, self.train_dataset, 19980)
        valid_numpy_labels, valid_numpy_labels_pred = Inference.create_label_and_pred(self, self.validation_dataset, 830)

        train_eer = Inference.eer(train_numpy_labels, train_numpy_labels_pred)
        valid_eer = Inference.eer(valid_numpy_labels, valid_numpy_labels_pred)
        
        print(f'EER value for training dataset is : {train_eer}')
        print(f'EER value for validation dataset is : {valid_eer}')


    def create_label_and_pred(self, dataset, dataset_size):

        numpy_images = np.zeros([dataset_size, 32, 32, 3])
        numpy_labels = np.zeros([dataset_size, 1])
        k = 0

        for images, labels in dataset.take(np.floor(dataset_size/self.batch_size) + 1):
            images_n = images.numpy()
            labels_n = labels.numpy()
            for i in range(labels_n.shape[0]):
                numpy_labels[i + k*self.batch_size] = labels_n[i]
                numpy_images[i + k*self.batch_size, :, :, :] = images_n[i]
            k = k + 1
        numpy_labels_pred = self.model.predict(numpy_images)

        return numpy_labels, numpy_labels_pred
    def scores(self, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):
        Inference.dataset(self)
        Inference.load_model(self,model_path=model_path)

        train_numpy_labels, train_numpy_labels_pred = Inference.create_label_and_pred(self, self.train_dataset, 19980)
        valid_numpy_labels, valid_numpy_labels_pred = Inference.create_label_and_pred(self, self.validation_dataset, 830)

        train_numpy_labels_pred = np.round(train_numpy_labels_pred, decimals=0)
        valid_numpy_labels_pred = np.round(valid_numpy_labels_pred, decimals=0)

        scores={}

        train_difference = train_numpy_labels_pred - train_numpy_labels
        valid_difference = valid_numpy_labels_pred - valid_numpy_labels
        training_overal_accuracy = 1 - np.sum(np.abs(train_difference))/train_numpy_labels.shape[0]
        valid_overal_accuracy = 1 - np.sum(np.abs(valid_difference))/valid_numpy_labels.shape[0]

        train_cache = np.array([0, 0, 0, 0])

        for i in range(train_difference.shape[0]):
            if train_difference[i] == 1:
                train_cache[0] = train_cache[0] + 1
            elif train_difference[i] == -1:
                train_cache[1] = train_cache[1] + 1
            else:
                if train_numpy_labels_pred[i] == 0:
                    train_cache[2] = train_cache[2] + 1
                else:
                    train_cache[3] = train_cache[3] + 1
        
        valid_cache = np.array([0, 0, 0, 0])

        for i in range(valid_difference.shape[0]):
            if valid_difference[i] == 1:
                valid_cache[0] = valid_cache[0] + 1
            elif valid_difference[i] == -1:
                valid_cache[1] = valid_cache[1] + 1
            else:
                if valid_numpy_labels_pred[i] == 0:
                    valid_cache[2] = valid_cache[2] + 1
                else:
                    valid_cache[3] = valid_cache[3] + 1

        scores['training overal accuracy'] = training_overal_accuracy
        scores['validation overal accuracy'] = valid_overal_accuracy
        scores['training alphabet accuracy'] = train_cache[2]/(train_cache[2] + train_cache[0])
        scores['training numeric accuracy'] = train_cache[3]/(train_cache[1] + train_cache[3])
        scores['validation alphabet accuracy'] = valid_cache[2]/(valid_cache[2] + valid_cache[0])
        scores['validation numeric accuracy'] = valid_cache[3]/(valid_cache[1] + valid_cache[3])

        score_df = pd.DataFrame(scores)

        return score_df

    def getScores(self, images_path, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):
        preds = Inference.predict(self, images_path=images_path, model_path=model_path)
        numeric_preds = preds
        alphabetic_preds = 1 - preds
        scores = {'Alphabetic probability' : list(alphabetic_preds), 'Numeric probability' : list(numeric_preds)}
        scores_df = pd.DataFrame(scores)
        scores_df.to_html("scores.html")
        return scores_df