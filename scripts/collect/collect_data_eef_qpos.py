# -- coding: UTF-8
import os
import argparse

from rhos_cobot.data_collection import RosOperator, save_data


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
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
    # parser.add_argument('--use_depth_image', action='store_true', help='use_depth_image')
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=True, required=False)

    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=25, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions, actions_eef, exit_type = ros_operator.process()

    # 根据退出类型选择保存目录
    if exit_type == 'e':
        # 按e键退出，保存到失败数据目录
        print("Saving failed demonstration data...")
        dataset_dir = "./fail_data"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(
            dataset_dir, args.task_name + "_episode_" + str(args.episode_idx))
    else:
        # 正常退出或按q键退出，保存到正常数据目录
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)

        # if(len(actions) < args.max_timesteps):
        #     print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
        #     exit(-1)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(
            dataset_dir, "episode_" + str(args.episode_idx))

    save_data(args, timesteps, actions, actions_eef, dataset_path)
    print(f"use_depth_img: {args.use_depth_image}")


if __name__ == '__main__':
    main()
