

import os
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.num_gpu = 1
config.TRAIN.process_num = 2               ####process num
config.TRAIN.prefetch_size = 60               ####Q size for data provider

config.TRAIN.batch_size = 64
config.TRAIN.log_interval = 100
config.TRAIN.epoch = 300

config.TRAIN.lr_value_every_epoch = [0.00001,0.0001,0.001,0.0001,0.00001,0.000001]          ####lr policy
config.TRAIN.lr_decay_every_epoch = [1,2,120,160,200]
config.TRAIN.weight_decay_factor = 5.e-4                                    ###########l2
config.TRAIN.mix_precision=False                                            ##use mix precision to speedup
config.TRAIN.vis= False
config.TRAIN.opt= 'Adam'


config.MODEL = edict()

# config.MODEL.model_path = './model'                         # save directory
# config.MODEL.checkpoints_path = './checkpoints/my_checkpoints'
# config.MODEL.pretrained_model= './checkpoints/my_checkpoints'                                     ######

config.MODEL.model_path = '/content/drive/My Drive/model'                                    # save directory
config.MODEL.checkpoints_path = '/content/drive/My Drive/checkpoints/my_checkpoints'
config.MODEL.pretrained_model= '/content/drive/My Drive/checkpoints/my_checkpoints'                                     ######

#####
config.MODEL.hin = 256                                                  # input size during training , 512  different with the paper
config.MODEL.win = 256
config.MODEL.num_classes = 3
# config.MODEL.feature_maps_size=[[32,32],[16,16],[8,8]]
config.MODEL.feature_maps_size=[[32, 32]]
# config.MODEL.num_anchors=21824  ##it should be
config.MODEL.num_anchors=21504

config.MODEL.MATCHING_THRESHOLD = 0.35
config.MODEL.max_negatives_per_positive= 3.0

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from lib.core.model.facebox.anchor_generator import AnchorGenerator

anchorgenerator = AnchorGenerator()
config.MODEL.anchors=anchorgenerator(config.MODEL.feature_maps_size, (config.MODEL.hin, config.MODEL.win))

config.TEST = edict()
config.TEST.score_threshold=0.05
config.TEST.iou_threshold=0.3
config.TEST.max_boxes=100
config.TEST.parallel_iterations=8

config.DATA = edict()
config.DATA.root_path=''
# config.DATA.train_txt_path='train.txt'
# config.DATA.val_txt_path='val.txt'
config.DATA.train_txt_path='/content/trafficsign_faceboxes_tensorflow_2/train_colab.txt'
config.DATA.val_txt_path='/content/trafficsign_faceboxes_tensorflow_2/val_colab.txt'

config.DATA.cover_small_face = 50                      ##small faces are covered
############NOW the model is trained with RGB mode
config.DATA.PIXEL_MEAN = [123., 116., 103.]   ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]











