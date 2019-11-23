from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
from .augmentation import augment_seg
import random
random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]
from .data_loader import get_pairs_from_paths, get_image_arr, get_segmentation_arr, verify_segmentation_dataset
import math

IMAGE_ORDERING = 'channels_last'

class DataSequence(Sequence):

    def __init__(self, images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False):
        """
        Keras Sequence object to train a model on larger-than-memory data.
            @:param: data_dir: directory in which we have got the kitti images and the corresponding masks
            @:param: batch_size: define the number of training samples to be propagated.
            @:param: image_shape: shape of the input image
        """

        self.images_path = images_path
        self.segs_path = segs_path
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.do_augment = do_augment
        self.img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

       
    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.img_seg_pairs) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        X = []
        Y = []
        img_seg_pairs_batch = self.img_seg_pairs[idx * self.batch_size: (1 + idx) * self.batch_size]
        for im , seg in img_seg_pairs_batch:

            im = cv2.imread(im , 1 )
            seg = cv2.imread(seg , 1 )

            if self.do_augment:
                im , seg[:,:,0] = augment_seg( im , seg[:,:,0] )

            X.append( get_image_arr(im , self.input_width , self.input_height ,ordering=IMAGE_ORDERING )  )
            Y.append( get_segmentation_arr( seg , self.n_classes , self.output_width , self.output_height )  )

        return np.array(X) , np.array(Y)
