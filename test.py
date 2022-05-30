from Inference_Class import Inference
import cv2
import numpy as np
import glob

def image_to_list(images_path, type='jpg'):


    images_list = glob.glob(f'{images_path}/*.{type}')
    images = []
    for i in range(len(images_list)):
        image = cv2.imread(images_list[i])
        image_array = np.asarray(image)
        images.append(image_array)

    return images
images = image_to_list('model/valid/numeric')
print(Inference('tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)').getScores(images=images))