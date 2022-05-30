from Inference_Class import Inference
from train import Train
import tensorflow as tf
import cv2
import os
import numpy as np
import glob
#print(Inference("E:/Educate/Work/model").predict('../Datasets/work/fashion/valid/numeric/',model_weights_path='E:/Educate/Work/model/tmp/checkpoint_alpha=0.1_bn_aug(rc)_do(1)'))
preds = Inference('tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)').getScores(images_path='model/valid/numeric')
def image_to_list(images_path, type='jpg'):


    images_list = glob.glob(f'{images_path}/*.{type}')
    images = []
    for i in range(len(images_list)):
        image = cv2.imread(images_list[i])
        image_array = np.asarray(image)
        images.append(image_array)

    return images
images = image_to_list('model/valid/numeric')
preds2 = Inference('tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)').getScores(images=images)
print(preds==preds2)