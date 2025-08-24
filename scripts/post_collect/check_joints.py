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
    """
    加载HDF5文件并返回所需数据
    只加载关节相关的数据以节省内存和时间
    只返回qpos和action
    失败时返回None
    失败时会通过logging记录错误
    """
    if not os.path.isfile(file_path):
        logger.error(f'Dataset does not exist at {file_path}')
        return None

    try:
        with h5py.File(file_path, 'r') as root:
            compressed = root.attrs.get('compress', False)
            if not compressed:
                logger.warning(
                    f"Data in file {file_path} is not compressed. Please ensure this is expected.")
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            action_eef = root['/action_eef'][()
                                             ] if 'action_eef' in root else None

        return qpos, action
    except Exception as e:
        logger.error(f"Failed to load HDF5 file {file_path}: {e}")
        return None


def check_constant_or_empty(data, data_key, file_path):
    """
    检查时序数据是否全是一个固定值或者为空。
    如果是，则通过logging记录警告。
    """
    if data is None or data.size == 0:
        logging.warning(f"Data for '{data_key}' in file {file_path} is empty.")
        return True

    # 判断数据是否几乎全是一个固定值
    # 这里我们计算数据的标准差，如果接近0，则认为它是常数
    # 使用 np.isclose 来处理浮点数比较的精度问题
    std_dev = np.std(data)
    if np.isclose(std_dev, 0, atol=1e-6):  # 可以根据需要调整容忍度 atol
        logging.warning(
            f"Data for '{data_key}' in file {file_path} is constant. For example, {data[0]}")
        return True

    return False


def check_sudden_changes(data, data_key, file_path, threshold=0.5):
    """
    检查时序数据中是否存在突变（即相邻时间步之间的变化过大）。
    这里使用了一个简单的阈值方法。
    """
    if data is None or data.size == 0 or data.ndim < 2:
        return

    # 计算相邻时间步长之间的绝对差值
    # np.diff() 函数返回一个由相邻元素差值组成的数组
    # 我们只关心第一个维度（时间维度）的变化
    time_diffs = np.abs(np.diff(data, axis=0))

    # 找到任何一个差值超过阈值的位置
    if np.any(time_diffs > threshold):
        outlier_indices = np.where(time_diffs > threshold)
        logging.warning(
            f"Sudden change detected in file: {file_path}. Data key '{data_key}' has values changing by more than {threshold}.")
        logging.warning(f"Outlier indices: {outlier_indices}")


def check_pi_outliers(data, data_key, file_path):
    """
    检查时序数据中是否有任何值超出 [-pi, pi] 范围。
    如果发现问题，会通过logging记录下来。
    """
    if np.any(np.abs(data) > np.pi):
        # 找到所有超出范围的索引
        outlier_indices = np.where(np.abs(data) > np.pi)
        outlier_values = data[outlier_indices]
        logging.warning(
            f"Outlier found in file: {file_path}. Data key '{data_key}' has values outside [-pi, pi] range.")
        logging.warning(
            f"Outlier indices: {outlier_indices}, Outlier values: {outlier_values}")
        return True
    return False


def check_file(file_path, data_key):
    """
    检查单个HDF5文件中的数据。
    - 检查数据是否为空或为常数。
    - 检查数据是否在 [-pi, pi] 范围内。
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
            logging.error(
                f"Data key '{data_key}' is not valid. Choose from {list(data_map.keys())}")
            return

        target_data = data_map.get(data_key)

        # 在范围检查之前，先检查数据是否为空或为常数
        if check_constant_or_empty(target_data, data_key, file_path):
            return

        if target_data.ndim < 2:
            logging.warning(
                f"Invalid data shape for '{data_key}' in file {file_path}. Skipping.")
            return

        # 核心检查逻辑：如果数据中有任何一个绝对值超过pi，则记录警告
        if check_pi_outliers(target_data, data_key, file_path):
            return
        # 检查是否有突变
        check_sudden_changes(target_data, data_key, file_path)

    except Exception as e:
        logging.error(
            f"An unexpected error occurred while processing {file_path}: {e}")


def main(args):
    setup_logging(args)

    dataset_dir = args.dataset_dir
    data_key = args.data_key
    max_workers = args.max_workers
    task_name = args.task_name

    # 获取所有任务名称
    if task_name:
        task_list = [task_name]
    else:
        task_list = [d for d in os.listdir(
            dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        task_list.sort()

    # 1. 收集所有需要检查的文件路径
    all_episode_files = []
    for task_name in task_list:
        source_dir = os.path.join(dataset_dir, task_name)
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    all_episode_files.append(os.path.join(root, file))

    all_episode_files.sort()

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
        f"All files have been checked. See {os.path.join(args.dataset_dir, 'check_log', f'check_{args.data_key}.log')} for details.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Detect data outliers in a robot dataset using multithreading.")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root dataset directory.')
    parser.add_argument('--data_key', type=str, required=True,
                        help="Data key to process (e.g., 'qpos', 'action').")
    parser.add_argument('--task_name', type=str, default=None,
                        help='Specific task name to check. If not provided, all tasks will be checked.')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(),
                        help='Maximum number of threads to use. Defaults to the number of CPU cores.')

    args = parser.parse_args()
    main(args)
