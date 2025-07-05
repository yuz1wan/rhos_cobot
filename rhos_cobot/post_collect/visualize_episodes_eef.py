#coding=utf-8
import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
from .utils import load_hdf5

DT = 0.02
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]
EEF_NAMES = [
    "pos_x", "pos_y", "pos_z",                      # 末端执行器的位置
    "orient_x", "orient_y", "orient_z", "orient_w", # 末端执行器的方向
    "joint_last",                                   # 最后一个关节位置
]

# def load_hdf5(dataset_dir, dataset_name):
#     dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
#     if not os.path.isfile(dataset_path):
#         print(f'Dataset does not exist at \n{dataset_path}\n')
#         exit()

#     with h5py.File(dataset_path, 'r') as root:
#         is_sim = root.attrs['sim']
#         compressed = root.attrs.get('compress', False)
#         qpos = root['/observations/qpos'][()]
#         qvel = root['/observations/qvel'][()]
#         if 'effort' in root.keys():
#             effort = root['/observations/effort'][()]
#         else:
#             effort = None
#         action = root['/action'][()]
#         action_eef = root['/action_eef'][()]
#         base_action = root['/base_action'][()]
#         image_dict = dict()
#         image_dict_depth = dict()
#         for cam_name in root[f'/observations/images/'].keys():
#             image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
#         # for cam_name in root[f'/observations/images_depth/'].keys():
#         #     image_dict_depth[cam_name] = root[f'/observations/images_depth/{cam_name}'][()]

#     if compressed:
#         for cam_id, cam_name in enumerate(image_dict.keys()):
#             # un-pad and uncompress
#             padded_compressed_image_list = image_dict[cam_name]
#             image_list = []
#             for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): 
#                 image = cv2.imdecode(np.frombuffer(padded_compressed_image, np.uint8), cv2.IMREAD_COLOR)
#                 image_list.append(image)
#             image_dict[cam_name] = image_list

#     return qpos, qvel, effort, action, action_eef, base_action, image_dict, image_dict_depth

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    task_name   = args['task_name']
    dataset_name = f'episode_{episode_idx}'
    display = args.get('display', True)

    visualize_dir = os.path.join(dataset_dir, "./Visualization", task_name)
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    qpos, qvel, effort, action, action_eef, base_action, image_dict, _ = load_hdf5(os.path.join(dataset_dir, task_name), dataset_name)
    
    print('hdf5 loaded!!')
    
    save_videos(image_dict, action, DT,  video_path=os.path.join(visualize_dir, dataset_name + '_video.mp4'), display=display)
    # save_depth_videos(image_depth_dict, action, DT,  video_path=os.path.join(dataset_dir, dataset_name + 'depth_video.mp4'))
   
    visualize_joints(qpos, action, plot_path=os.path.join(visualize_dir, dataset_name + '_qpos.png'))
    visualize_base(base_action, plot_path=os.path.join(visualize_dir, dataset_name + '_base_action.png'))
    visualize_eef(action_eef,plot_path=os.path.join(visualize_dir, dataset_name + '_eef.png'))

def save_videos(video, actions, dt, video_path=None, display=True):
    cam_names = list(video.keys())
    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(video[cam_name])
    all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]
        image = image[:, :, [2, 1, 0]]  # swap B and R channel
        if display:
            cv2.putText(image, f"Frame: {t+1}/{n_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Left: {np.round(actions[t][:7], 3)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f"Right: {np.round(actions[t][7:], 3)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("images",image)
            cv2.waitKey(30)
        # print("frame_id: ", t, "left: ", np.round(actions[t][:7], 3), "right: ", np.round(actions[t][7:], 3))
        out.write(image)
    out.release()
    print(f'Saved video to: {video_path}')

def save_depth_videos(video, actions, dt, video_path='depth_video.mp4'):
    # 指定你想保存的深度图相机名称
    cam_names = list(video.keys())

    all_cam_videos = []

    for cam_name in cam_names:
        decoded_video = []
        for encoded in video[cam_name]:
            # 解码为单通道图像
            img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)

            # 归一化并转换为uint8（如果不是）
            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)

            # 伪彩色可视化（可选），否则注释掉这行显示灰度图
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            decoded_video.append(img)
        all_cam_videos.append(decoded_video)

    # 拼接成 (N, H, W_total, 3)
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)

    n_frames, h, w, _ = np.array(all_cam_videos).shape
    fps = int(1 / dt)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for t in range(n_frames):
        image = all_cam_videos[t]
        cv2.imshow("depth_video", image)
        cv2.waitKey(int(1000 * dt))
        print("episode_id: ", t,
              "left: ", np.round(actions[t][:7], 3),
              "right: ", np.round(actions[t][7:], 3), "\n")
        out.write(image)

    out.release()
    cv2.destroyAllWindows()
    print(f"Saved depth video to: {video_path}")

def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1, color='orangered')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    # for dim_idx in range(num_dim):
    #     ax = axs[dim_idx]
    #     ax.plot(command[:, dim_idx], label=label2)
    #     ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_base(readings, plot_path=None):
    readings = np.array(readings) # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()


    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()

def visualize_eef(readings, plot_path=None):
    readings = np.array(readings) # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot eef
    all_names = ['left_' + name for name in EEF_NAMES] + ['right_' + name for name in EEF_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw_data')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        # ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'eef {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved eef plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',default=0, required=False)
    parser.add_argument('--display', action='store_true', help='Display the video frames. (If use in headless terminal, dont set this)', required=False)
    
    main(vars(parser.parse_args()))
