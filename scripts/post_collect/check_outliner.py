# coding=utf-8
import os
import argparse
import numpy as np
import h5py
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def setup_logging(args):
    """配置日志记录，同时输出到文件和控制台"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    LOGGING_PATH = os.path.join(
        args.dataset_dir, "check_log", f'check_{args.data_key}.log')
    if not os.path.exists(os.path.dirname(LOGGING_PATH)):
        print(f"Make dir {os.path.dirname(LOGGING_PATH)}")
        os.makedirs(os.path.dirname(LOGGING_PATH))
    else:
        print(f"Has dir {os.path.dirname(LOGGING_PATH)}")

    # 创建文件处理器
    file_handler = logging.FileHandler(
        LOGGING_PATH, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# 获取一个logger实例
logger = logging.getLogger(__name__)


def load_hdf5_joints(file_path):
    """加载HDF5文件并返回所需数据"""
    if not os.path.isfile(file_path):
        logger.error(f'Dataset does not exist at {file_path}')
        return None

    try:
        with h5py.File(file_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            action_eef = root['/action_eef'][()
                                             ] if 'action_eef' in root else None

        return qpos, action
    except Exception as e:
        logger.error(f"Failed to load HDF5 file {file_path}: {e}")
        return None


def check_file(file_path, data_key):
    """
    检查单个HDF5文件中的数据是否在[-pi, pi]范围内。
    如果发现问题，会通过logging记录下来。
    """
    try:
        loaded_data = load_hdf5_joints(file_path)
        if loaded_data is None:
            return  # 加载失败，日志已在load_hdf5中记录

        qpos, action = loaded_data

        data_map = {
            'qpos': qpos,
            'action': action,
        }

        if data_key not in data_map:
            logger.error(
                f"Data key '{data_key}' is not valid. Choose from {list(data_map.keys())}")
            return

        target_data = data_map.get(data_key)

        if target_data is None:
            logger.info(
                f"Data key '{data_key}' not present in {file_path}. Skipping check.")
            return

        if target_data.ndim < 2:
            logger.warning(
                f"Invalid data shape for '{data_key}' in file {file_path}. Skipping.")
            return

        # 核心检查逻辑：如果数据中有任何一个绝对值超过pi，则记录警告
        if np.any(np.abs(target_data) > np.pi):
            # 找到所有超出范围的索引
            outlier_indices = np.where(np.abs(target_data) > np.pi)
            outlier_values = target_data[outlier_indices]
            logger.warning(
                f"Outlier found in file: {file_path}. Data key '{data_key}' has values outside [-pi, pi] range.")
            logger.warning(
                f"Outlier indices: {outlier_indices}, Outlier values: {outlier_values}")

    except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing {file_path}: {e}")


def main(args):
    """主函数，负责收集文件并使用线程池进行处理"""
    setup_logging(args)

    dataset_dir = args.dataset_dir
    data_key = args.data_key
    max_workers = args.max_workers

    task_list = [d for d in os.listdir(
        dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not task_list:
        logger.info(f"No tasks found in {dataset_dir}")
        return
    logger.info(f"Found {len(task_list)} tasks: {task_list}")

    # 1. 收集所有需要检查的文件路径
    all_episode_files = []
    for task_name in task_list:
        source_dir = os.path.join(dataset_dir, task_name)
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    all_episode_files.append(os.path.join(root, file))

    if not all_episode_files:
        logger.info("No .hdf5 files found across all tasks.")
        return

    logger.info(
        f"Found a total of {len(all_episode_files)} episodes to check across all tasks.")

    # 2. 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用tqdm创建进度条
        list(tqdm(executor.map(lambda f: check_file(f, data_key), all_episode_files),
                  total=len(all_episode_files),
                  desc=f"Checking '{data_key}'"))

    logger.info(
        "All files have been checked. See 'outlier_detect.log' for details.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect data outliers in a robot dataset using multithreading.")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root dataset directory.')
    parser.add_argument('--data_key', type=str, required=True,
                        help="Data key to process (e.g., 'qpos', 'action').")
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(),
                        help='Maximum number of threads to use. Defaults to the number of CPU cores.')

    args = parser.parse_args()
    main(args)
