from Inference_Class import Inference
#print(Inference("E:/Educate/Work/model").predict('../Datasets/work/fashion/valid/numeric/',model_weights_path='E:/Educate/Work/model/tmp/checkpoint_alpha=0.1_bn_aug(rc)_do(1)'))
Inference("model/").getScores(images_path='model/valid/alphabet',model_path='tmp/models/checkpoint_alpha=0.01_bn_aug(rc)_do(1)')