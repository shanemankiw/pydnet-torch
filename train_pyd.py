import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from main_monodepth_pytorch import Model

dict_parameters = edict({'data_dir':'../kitti/',
                         'val_data_dir':'../kitti_val/',
                         'model_path':'./saved_models/monodepth_pyd_001.pth',
                         'output_directory':'./output_pics/',
                         'input_height':256,
                         'input_width':512,
                         'model':'pydnet',
                         'pretrained':True,
                         'mode':'train',
                         'epochs':200,
                         'learning_rate':1e-4,
                         'batch_size': 32,
                         'adjust_lr':True,
                         'device':'cuda:0',
                         'do_augmentation':True,
                         'augment_parameters':[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                         'print_images':False,
                         'print_weights':False,
                         'input_channels': 3,
                         'num_workers': 8,
                         'use_multiple_gpu': False})

model = Model(dict_parameters)

model.train()