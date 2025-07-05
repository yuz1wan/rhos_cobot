# -- coding: UTF-8
from .collect_data_eef import RosOperator
from .collect_data_eef import save_data
import os
import time
import signal
import numpy as np
import h5py
import argparse
import dm_env

import collections
from collections import deque

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2

def get_dataset_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_num', action='store', type=int, help='Episode Number.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=500, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    #  topic name of color image
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    # topic name of depth image
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    # topic name of arm
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    # topic name of eef
    parser.add_argument('--puppet_eef_left_topic', action='store', type=str, help='puppet_eef_left_topic',
                        default='/puppet/end_pose_left', required=False)
    parser.add_argument('--puppet_eef_right_topic', action='store', type=str, help='puppet_eef_right_topic',
                        default='/puppet/end_pose_right', required=False)
    
    # topic name of robot_base
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    # collect depth image
    parser.add_argument('--use_depth_image', action='store_true', help='use_depth_image',
                        required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=25, required=False)
    
    args = parser.parse_args()
    return args

def main():
    signal.signal(signal.SIGINT, lambda:sys.exit(0)) # exit when press Ctrl C

    args = get_dataset_arguments()
    # args.use_depth_image = False
    ros_operator = RosOperator(args)
    
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    
    print(f"use_depth_img: {args.use_depth_image}")
    num_episode = args.episode_num
    episode_idx = 0
    failed_idx = 0
    while episode_idx < num_episode:
        if os.path.exists(os.path.join(dataset_dir, "episode_" + str(episode_idx) + ".hdf5")):
            episode_idx+=1
            continue

        print("======= Collecting data for episode {} =======".format(episode_idx))
        timesteps, actions ,actions_eef= ros_operator.process()

        if(len(actions) < args.max_timesteps):
            print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
            continue
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        print("======= Collecting Done for episode {} =======".format(episode_idx))

        
        # user press 'q' to stop, or 'c' to continue
        key = input("Press 'c' to continue, 'r' to re-record, 'q' to quit, 'f' to record fail case: ")
        if key == 'r':
            print("Re-recording...")
            continue
        elif key == 'c':
            print("Saving and Continuing...")
            # save data
            data_path = os.path.join(dataset_dir, "episode_" + str(episode_idx))
            while os.path.exists(f"{data_path}.hdf5"):
                episode_idx += 1
                data_path = os.path.join(dataset_dir, "episode_" + str(episode_idx))
            save_data(args, timesteps, actions, actions_eef, data_path)
            print("======= Episode {} data saved =======".format(episode_idx))
            print()
            episode_idx += 1
            
        elif key == 'q':
            print("Saving and Quitting...")
            # save data
            data_path = os.path.join(dataset_dir, "episode_" + str(episode_idx))
            save_data(args, timesteps, actions, actions_eef, data_path)
            print("======= Episode {} data saved =======".format(episode_idx))
            print()
            episode_idx += 1
            return
        
        elif key == 'f':
            print("This Episode Failed! Saving and Re-Record!")
            # save failed data
            failed_dir = os.path.join(dataset_dir, "failed_episodes")
            if not os.path.exists(failed_dir):
                os.makedirs(failed_dir)
            data_path = os.path.join(failed_dir, "episode_" + str(failed_idx))
            while os.path.exists(f"{data_path}.hdf5"):
                failed_idx += 1
                data_path = os.path.join(failed_dir, "episode_" + str(failed_idx))
            save_data(args, timesteps, actions, actions_eef, data_path)
            print("======= Failed Episode {} data saved =======".format(failed_idx))
            print()
            failed_idx += 1
        else:
            print("Invalid input, Defaulting to continue...")
            continue
            
        
        
        
    print("All episodes collected.")
    
if __name__ == "__main__":
    main()

# cd collect_data
# python collect_dataset_eef.py --dataset_dir ./data --task_name stir --max_timesteps 500 --episode_num 10