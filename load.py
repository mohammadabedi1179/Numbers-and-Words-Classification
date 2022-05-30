from train import Train
import tensorflow as tf
import numpy as np
import sklearn.metrics
import pandas as pd
import glob
import cv2
import os
import shutil

class Load():
    def __init__(self):
        pass

    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        self.model = model
        return model


    def eer(numpy_labels, numpy_labels_pred, positive_label=1):

        fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy_labels, numpy_labels_pred, pos_label=positive_label)
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        eer = eer_1/2 + eer_2/2

        return eer
    
    def compute_eer(self, model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)'):
        Train("/model").dataset(self)
        Load.load_model(self, model_path=model_path)

        train_numpy_labels, train_numpy_labels_pred = Load.create_label_and_pred(self, self.train_dataset, 19980)
        valid_numpy_labels, valid_numpy_labels_pred = Load.create_label_and_pred(self, self.validation_dataset, 830)

        train_eer = Load.eer(train_numpy_labels, train_numpy_labels_pred)
        valid_eer = Load.eer(valid_numpy_labels, valid_numpy_labels_pred)
        
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
        Train('/model').dataset(self)
        Load.load_model(self,model_path=model_path)

        train_numpy_labels, train_numpy_labels_pred = Load.create_label_and_pred(self, self.train_dataset, 19980)
        valid_numpy_labels, valid_numpy_labels_pred = Load.create_label_and_pred(self, self.validation_dataset, 830)

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

    def preprocess_image_path(self, images_path, type='jpg'):

        images_list = glob.glob(f'{images_path}/*.{type}')
        images = np.zeros([len(images_list), 32, 32, 3])
        for i in range(len(images_list)):
            image = cv2.imread(images_list[i])
            if image.shape != (32, 32, 3):
                image = cv2.resize(image, (32, 32))
            image_array = np.asarray(image)
            images[i, :, :, :] = image_array

        return images
    
    def preprocess_image(self, images_list):
        
        images = np.zeros([len(images_list), 32, 32, 3])
        os.mkdir('tmp/images')
        for i in range(len(images_list)):
            cv2.imwrite(f'tmp/images/{i}.jpg', images_list[i])
            image = cv2.imread(f'tmp/images/{i}.jpg')

            if image.shape != (32, 32, 3):
                image = cv2.resize(image, (32, 32))
            image_array = np.asarray(image)
            images[i, :, :, :] = image_array
        shutil.rmtree('tmp/images')

        return images