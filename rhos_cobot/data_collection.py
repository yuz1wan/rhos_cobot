# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env

import collections
from collections import deque

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import sys
import cv2
import keyboard
import select


def check_keypress():
    """非阻塞监听终端键盘输入"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def images_encoding(imgs, mode='color'):
    encode_data = []
    padded_data = []
    max_len = 0

    for img in imgs:
        if mode == 'color':
            success, encoded = cv2.imencode('.jpg', img)
        elif mode == 'depth':
            success, encoded = cv2.imencode('.png', img)  # PNG 支持 uint16
        else:
            raise ValueError("Unsupported mode, must be 'color' or 'depth'")

        if not success:
            raise RuntimeError("Image encoding failed")

        byte_data = encoded.tobytes()
        encode_data.append(byte_data)
        max_len = max(max_len, len(byte_data))

    # # padding
    # padded_data = [data.ljust(max_len, b'\0') for data in encode_data]  # 更简洁的写法

    return encode_data, max_len

# 保存数据函数


def save_data(args, timesteps, actions, actions_eef, dataset_path):
    # 数据字典
    data_size = len(actions)
    data_dict = {
        # 一个是奖励里面的qpos，qvel， effort ,一个是实际发的acition
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/action_eef': [],  # size(16,)
        '/base_action': [],
        # '/base_action_t265': [],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        # 循环弹出一个队列
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧
        action_eef = actions_eef.pop(0)

        # 往字典里面添值
        # Timestep返回的qpos，qvel,effort
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])

        # 实际发的action
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        data_dict['/action_eef'].append(action_eef)

        # 相机数据
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(
                ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(
                    ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # 文本的属性：
        # 1 是否仿真
        # 2 图像是否压缩
        #
        root.attrs['sim'] = False
        root.attrs['compress'] = True

        # 创建一个新的组observations，观测状态组
        # 图像组
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            enc_imgs, max_len = images_encoding(
                data_dict[f'/observations/images/{cam_name}'], mode='color')
            data_dict[f'/observations/images/{cam_name}'] = enc_imgs
            dt = f'S{max_len}'
            image.create_dataset(cam_name, (data_size,), dtype=dt)

        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                enc_depths, max_len_depth = images_encoding(
                    data_dict[f'/observations/images_depth/{cam_name}'], mode='depth')
                data_dict[f'/observations/images_depth/{cam_name}'] = enc_depths
                dt_depth = f'S{max_len_depth}'
                image_depth.create_dataset(
                    cam_name, (data_size,), dtype=dt_depth)

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))
        _ = root.create_dataset('action_eef', (data_size, 16))

        # data_dict write into h5py.File
        for name, array in data_dict.items():
            root[name][...] = array
    print(
        f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.puppet_eef_right_deque = None
        self.puppet_eef_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_eef_left_deque = deque()
        self.puppet_eef_right_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(
            ), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_eef_left_deque) == 0 or self.puppet_eef_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_eef_right_deque) == 0 or self.puppet_eef_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(
            self.img_left_deque.popleft(), 'passthrough')
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(
            self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(
            self.img_front_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        while self.puppet_eef_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_eef_left_deque.popleft()
        puppet_eef_left = self.puppet_eef_left_deque.popleft()

        while self.puppet_eef_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_eef_right_deque.popleft()
        puppet_eef_right = self.puppet_eef_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(
                img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(
                self.img_right_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_right_depth = cv2.copyMakeBorder(
            img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_front_depth = cv2.copyMakeBorder(
            img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base, puppet_eef_left, puppet_eef_right)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def puppet_eef_left_callback(self, msg):
        if len(self.puppet_eef_left_deque) >= 2000:
            self.puppet_eef_left_deque.popleft()
        self.puppet_eef_left_deque.append(msg)

    def puppet_eef_right_callback(self, msg):
        if len(self.puppet_eef_right_deque) >= 2000:
            self.puppet_eef_right_deque.popleft()
        self.puppet_eef_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image,
                         self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image,
                         self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image,
                         self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image,
                             self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image,
                             self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image,
                             self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.args.master_arm_left_topic, JointState,
                         self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState,
                         self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState,
                         self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState,
                         self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_eef_left_topic, PoseStamped,
                         self.puppet_eef_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_eef_right_topic, PoseStamped,
                         self.puppet_eef_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry,
                         self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps = []
        actions = []
        actions_eef = []
        # 图像数据
        image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0

        # input_key = input("please input s:")
        # while input_key != 's' and not rospy.is_shutdown():
        #     input_key = input("please input s:")

        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True

        while not rospy.is_shutdown():
            if count >= self.args.max_timesteps + 1:
                break

            # 检测键盘按键（按 q 或 e 退出）
            exit_key = None
            if check_keypress():
                key = sys.stdin.readline().strip()
                if key == 'q':
                    print(
                        "\033[33m\n[INFO] Early stopping triggered by 'q' key.\033[0m\n")
                    exit_key = 'q'
                    break
                elif key == 'e':
                    print(
                        "\033[33m\n[INFO] Early stopping triggered by 'e' key.\033[0m\n")
                    exit_key = 'e'
                    break
            # 2 收集数据
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            count += 1
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base, puppet_eef_left, puppet_eef_right) = result
            # 2.1 图像信息
            image_dict = dict()
            image_dict[self.args.camera_names[0]] = img_front
            image_dict[self.args.camera_names[1]] = img_left
            image_dict[self.args.camera_names[2]] = img_right

            # 2.2 可视化三个视角的图像
            combined_image = np.concatenate((cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB), cv2.cvtColor(
                img_front, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)), axis=1)
            cv2.imshow('Combined View', combined_image)
            cv2.waitKey(1)

            # 2.2 从臂的信息从臂的状态 机械臂示教模式时 会自动订阅
            obs = collections.OrderedDict()  # 有序的字典
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = dict()
                image_dict_depth[self.args.camera_names[0]] = img_front_depth
                image_dict_depth[self.args.camera_names[1]] = img_left_depth
                image_dict_depth[self.args.camera_names[2]] = img_right_depth
                obs['images_depth'] = image_dict_depth
            obs['qpos'] = np.concatenate(
                (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate(
                (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate(
                (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            if self.args.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x,
                                   robot_base.twist.twist.angular.z]
            else:
                obs['base_vel'] = [0.0, 0.0]

            # 第一帧 只包含first， fisrt只保存StepType.FIRST
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                timesteps.append(ts)
                continue

            # 时间步
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 主臂保存状态
            actions.append(obs['qpos'])
            # eef
            action_eef = np.concatenate([np.array([puppet_eef_left.pose.position.x, puppet_eef_left.pose.position.y, puppet_eef_left.pose.position.z, ]), np.array([puppet_eef_left.pose.orientation.x, puppet_eef_left.pose.orientation.y, puppet_eef_left.pose.orientation.z, puppet_eef_left.pose.orientation.w,]), np.array(master_arm_left.position[-1:]),
                                         np.array([puppet_eef_right.pose.position.x, puppet_eef_right.pose.position.y, puppet_eef_right.pose.position.z]), np.array([puppet_eef_right.pose.orientation.x, puppet_eef_right.pose.orientation.y, puppet_eef_right.pose.orientation.z, puppet_eef_right.pose.orientation.w,]), np.array(master_arm_right.position[-1:])], axis=0)
            actions_eef.append(action_eef)
            timesteps.append(ts)
            print("Frame id: ", count)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()

        cv2.destroyAllWindows()
        actions = actions[1:] + actions[-1:]
        print("===================== WARNING ======================")
        print("In this version, we take next state of puppet arm as the action.")
        print("====================================================")
        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        print("len(actions_eef)  : ", len(actions_eef))

        # 返回退出类型，如果是按键退出则返回对应键值，否则返回'normal'
        return_exit_type = exit_key if exit_key else 'normal'
        return timesteps, actions, actions_eef, return_exit_type
