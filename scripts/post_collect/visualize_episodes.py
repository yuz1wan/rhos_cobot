# coding=utf-8
import os
import argparse
from rhos_cobot.utils import load_hdf5
from rhos_cobot.post_process import (
    save_videos, 
    visualize_joints, 
    visualize_base, 
    visualize_eef
)

DT = 0.02

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    task_name = args['task_name']
    dataset_name = f'episode_{episode_idx}'
    display = args.get('display', True)

    visualize_dir = os.path.join(dataset_dir, "./Visualization", task_name)
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    qpos, qvel, effort, action, base_action, image_dict, action_eef = load_hdf5(
        os.path.join(dataset_dir, task_name), dataset_name)

    print('hdf5 loaded!!')

    save_videos(image_dict, action, DT,  video_path=os.path.join(
        visualize_dir, dataset_name + '_video.mp4'), display=display)
    # save_depth_videos(image_depth_dict, action, DT,  video_path=os.path.join(dataset_dir, dataset_name + 'depth_video.mp4'))

    visualize_joints(qpos, action, plot_path=os.path.join(
        visualize_dir, dataset_name + '_qpos.png'))
    visualize_base(base_action, plot_path=os.path.join(
        visualize_dir, dataset_name + '_base_action.png'))
    visualize_eef(action_eef, plot_path=os.path.join(
        visualize_dir, dataset_name + '_eef.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store',
                        type=str, help='Dataset dir.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int,
                        help='Episode index.', default=0, required=False)
    parser.add_argument('--display', action='store_true',
                        help='Display the video frames. (If use in headless terminal, dont set this)', required=False)

    main(vars(parser.parse_args()))