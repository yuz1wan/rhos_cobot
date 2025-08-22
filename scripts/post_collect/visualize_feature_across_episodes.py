# coding=utf-8
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rhos_cobot.utils import load_hdf5 # 假设此函数在您的环境中可用

def visualize_feature_across_episodes(args):
    """
    遍历所有episode文件，加载指定数据，并绘制其某一维度的散点图和箱形图以寻找离群点。
    """
    dataset_dir = args['dataset_dir']
    task_name = args['task_name']
    data_key = args['data_key']
    feature_index = args['feature_index']

    # 构建数据源目录和可视化结果输出目录
    source_dir = os.path.join(dataset_dir, task_name)
    visualize_dir = os.path.join(dataset_dir, "./Visualization", task_name)
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)

    print(f"Scanning directory: {source_dir}")

    # 找到所有hdf5文件
    try:
        episode_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.hdf5')])
    except FileNotFoundError:
        print(f"Error: Directory not found at {source_dir}")
        return

    if not episode_files:
        print(f"No .hdf5 files found in {source_dir}")
        return

    print(f"Found {len(episode_files)} episodes. Processing...")
    
    # # 随机选取10个文件进行处理
    # import random
    # if len(episode_files) > 50:
    #     episode_files = random.sample(episode_files, 50)
    #     print(f"Randomly selected {len(episode_files)} episodes for visualization.")

    all_feature_data = []
    
    # 遍历所有文件并加载数据
    for filename in episode_files:
        dataset_name = os.path.splitext(filename)[0] # e.g., 'episode_0'
        try:
            # 加载HDF5文件中的所有数据
            qpos, qvel, effort, action, base_action, _, action_eef = load_hdf5(source_dir, dataset_name)
            
            # 使用一个字典来方便地通过key获取数据
            data_map = {
                'qpos': qpos,
                'qvel': qvel,
                'effort': effort,
                'action': action,
                'base_action': base_action,
                'action_eef': action_eef
            }

            if data_key not in data_map:
                print(f"Error: --data_key '{data_key}' is not valid. Choose from {list(data_map.keys())}")
                return
            
            target_data = data_map[data_key]

            # 检查数据维度是否有效
            if target_data is None or target_data.ndim < 2 or target_data.shape[1] <= feature_index:
                print(f"Warning: Invalid feature_index {feature_index} for data_key '{data_key}' in file {filename}. Shape is {target_data.shape}. Skipping.")
                continue

            # 提取指定维度的数据并添加到列表中
            feature_column = target_data[:, feature_index]
            all_feature_data.append(feature_column)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    if not all_feature_data:
        print("No data was loaded. Exiting.")
        return

    # 将列表中的所有数据连接成一个大的Numpy数组
    all_feature_data = np.concatenate(all_feature_data)
    print(f"Successfully aggregated {all_feature_data.shape[0]} data points.")

    # --- 开始绘图 ---
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Outlier Analysis for "{data_key}" - Feature Index {feature_index}\n(Task: {task_name})', fontsize=16)

    # 1. 绘制散点图
    axs[0].scatter(range(len(all_feature_data)), all_feature_data, s=5, alpha=0.6)
    axs[0].set_title('Scatter Plot of All Data Points')
    axs[0].set_xlabel('Data Point Index (Aggregated across all episodes)')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)

    # 2. 绘制箱形图
    axs[1].boxplot(all_feature_data, vert=False, flierprops=dict(marker='o', markersize=5, markerfacecolor='red'))
    axs[1].set_title('Box Plot for Outlier Detection')
    axs[1].set_xlabel('Value')
    axs[1].set_yticklabels(['']) # 隐藏Y轴标签
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应主标题
    
    # 保存图像
    plot_path = os.path.join(visualize_dir, f'agg_outlier_{data_key}_idx{feature_index}.png')
    plt.savefig(plot_path)
    print(f'Saved aggregated plot to: {plot_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a specific feature across all episodes to find outliers.")
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset directory.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--data_key', action='store', type=str, help="The key for the data to visualize (e.g., 'qpos', 'action').", required=True)
    parser.add_argument('--feature_index', action='store', type=int, help='The index of the feature/dimension to plot.', required=True)

    main_args = vars(parser.parse_args())
    visualize_feature_across_episodes(main_args)