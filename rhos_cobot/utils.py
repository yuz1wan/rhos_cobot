#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import rospy

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist

import sys

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        # if 'effort' in root.keys():
        #     effort = root['/observations/effort'][()]
        # else:
        #     effort = None
        effort = root['/observations/effort'][()] if 'effort' in root.keys() else None
        action = root['/action'][()]
        base_action = root['/base_action'][()] if 'base_action' in root.keys() else None
        action_eef = root['/action_eef'][()] if 'action_eef' in root.keys() else None
        
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    if compressed:
        print("This Data is compressed! Using Decoding!")
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                compressed_image = padded_compressed_image
                image = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
                image_list.append(image)
            image_dict[cam_name] = image_list

    return qpos, qvel, effort, action, base_action, image_dict, action_eef