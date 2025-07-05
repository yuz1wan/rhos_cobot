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
from geometry_msgs.msg import Twist, PoseStamped
from piper_msgs.msg import PosCmd
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from .utils import load_hdf5

def main(args):
    rospy.init_node("replay_node")
    bridge = CvBridge()
    img_left_publisher = rospy.Publisher(args.img_left_topic, Image, queue_size=10)
    img_right_publisher = rospy.Publisher(args.img_right_topic, Image, queue_size=10)
    img_front_publisher = rospy.Publisher(args.img_front_topic, Image, queue_size=10)
    
    puppet_arm_left_publisher = rospy.Publisher(args.puppet_arm_left_topic, JointState, queue_size=10)
    puppet_arm_right_publisher = rospy.Publisher(args.puppet_arm_right_topic, JointState, queue_size=10)
    
    master_arm_left_publisher = rospy.Publisher(args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(args.master_arm_right_topic, JointState, queue_size=10)

    puppet_eef_left_publisher = rospy.Publisher(args.puppet_eef_left_topic, PosCmd, queue_size=10)
    puppet_eef_right_publisher = rospy.Publisher(args.puppet_eef_right_topic, PosCmd, queue_size=10)
    
    robot_base_publisher = rospy.Publisher(args.robot_base_topic, Twist, queue_size=10)


    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    task_name   = args.task_name
    dataset_name = f'episode_{episode_idx}'

    origin_left = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279]  
    origin_right = [ 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
    endleft0 = [0.111376, -0.09219, 0.352633, 0.41211237354116054, 0.7627680187460423, -0.14576932099436934, 0.47654973109776155]
    endright0 =[0.305041, 0.18106, 0.339395, 0.23205404033907173, -0.9431655689825339, -0.11071169388959519, -0.21055296883269586]
# pose_left: 

    
    joint_state_msg = JointState()
    joint_state_msg.header =  Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
    twist_msg = Twist()

    eef_msg = PosCmd()
    # eef_msg.header = Header()

    rate = rospy.Rate(args.frame_rate)
    qposs, qvels, efforts, actions, base_actions, image_dicts, actions_eef= load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)
    
    
    if args.only_pub_master:
        last_action = [-0.0057,-0.031, -0.0122, -0.032, 0.0099, 0.0179, 0.2279, 0.0616, 0.0021, 0.0475, -0.1013, 0.1097, 0.0872, 0.2279]
        rate = rospy.Rate(100)
        for action in actions:
            if(rospy.is_shutdown()):
                    break
            
            new_actions = np.linspace(last_action, action, 20) # 插值
            last_action = action
            for act in new_actions:
                print(np.round(act[:7], 4))
                cur_timestamp = rospy.Time.now()  # 设置时间戳
                joint_state_msg.header.stamp = cur_timestamp 
                
                joint_state_msg.position = act[:7]
                master_arm_left_publisher.publish(joint_state_msg)

                joint_state_msg.position = act[7:]
                master_arm_right_publisher.publish(joint_state_msg)   

                if(rospy.is_shutdown()):
                    break
                rate.sleep() 
    
    else:
        i = 0
        
        while(not rospy.is_shutdown() and i < len(actions)):
            print("left: ", np.round(qposs[i][:7], 4))#, " right: ", np.round(qposs[i][7:], 4))
            
            cam_names = [k for k in image_dicts.keys()]
            image0 = image_dicts[cam_names[0]][i] 
            image0 = cv2.imdecode(np.frombuffer(image0, np.uint8), cv2.IMREAD_COLOR)
            image0 = image0[:, :, [2, 1, 0]]  # swap B and R channel
        
            image1 = image_dicts[cam_names[1]][i] 
            image1 = cv2.imdecode(np.frombuffer(image1, np.uint8), cv2.IMREAD_COLOR)
            image1 = image1[:, :, [2, 1, 0]]  # swap B and R channel
            
            image2 = image_dicts[cam_names[2]][i] 
            image2 = cv2.imdecode(np.frombuffer(image2, np.uint8), cv2.IMREAD_COLOR)
            image2 = image2[:, :, [2, 1, 0]]  # swap B and R channel

            cur_timestamp = rospy.Time.now()  # 设置时间戳
            
            # joint_state_msg.header.stamp = cur_timestamp 
            # joint_state_msg.position = actions[i][:7]
            # master_arm_left_publisher.publish(joint_state_msg)

            # joint_state_msg.position = actions[i][7:]
            # master_arm_right_publisher.publish(joint_state_msg)

            # joint_state_msg.position = qposs[i][:7]
            # puppet_arm_left_publisher.publish(joint_state_msg)

            # joint_state_msg.position = qposs[i][7:]
            # puppet_arm_right_publisher.publish(joint_state_msg)
            # eef_msg.header.stamp = cur_timestamp
            eef_msg.x, eef_msg.y, eef_msg.z = actions_eef[i][8:11]
            eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(actions_eef[i][11:15])
            eef_msg.gripper = actions_eef[i][15:16]
            puppet_eef_right_publisher.publish(eef_msg)

            eef_msg.x, eef_msg.y, eef_msg.z = actions_eef[i][:3]
            eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(actions_eef[i][3:7])
            eef_msg.gripper = actions_eef[i][7:8]
            puppet_eef_left_publisher.publish(eef_msg)

            # eef_msg.x, eef_msg.y, eef_msg.z = actions[i][8:11]
            # eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(np.concatenate([actions[i][11:14],actions[i][14:15]]))
            # eef_msg.gripper = actions[i][15:16]
            # puppet_eef_right_publisher.publish(eef_msg)

            # eef_msg.x, eef_msg.y, eef_msg.z = endleft0[:3]
            # eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(endleft0[3:7])
            # eef_msg.gripper = actions[i][7:8]
            # puppet_eef_left_publisher.publish(eef_msg)

            # eef_msg.x, eef_msg.y, eef_msg.z = actions[i][8:11]
            # eef_msg.roll, eef_msg.pitch, eef_msg.yaw = euler_from_quaternion(np.concatenate([actions[i][14:15], actions[i][11:14]]))
            # eef_msg.gripper = actions[i][15:16]
            # puppet_eef_right_publisher.publish(eef_msg)

            img_front_publisher.publish(bridge.cv2_to_imgmsg(image0, "bgr8"))
            img_left_publisher.publish(bridge.cv2_to_imgmsg(image1, "bgr8"))
            img_right_publisher.publish(bridge.cv2_to_imgmsg(image2, "bgr8"))
    
            
            twist_msg.linear.x = base_actions[i][0]
            twist_msg.angular.z = base_actions[i][1]
            robot_base_publisher.publish(twist_msg)

            i += 1
            rate.sleep() 
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)

    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=0, required=False)
    
    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--puppet_eef_left_topic', action='store', type=str, help='puppet_eef_left_topic',
                        default='/pos_cmd_left', required=False)
    parser.add_argument('--puppet_eef_right_topic', action='store', type=str, help='puppet_eef_right_topic',
                        default='/pos_cmd_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    
    parser.add_argument('--only_pub_master', action='store_true', help='only_pub_master',required=False)
    
    

    args = parser.parse_args()
    main(args)