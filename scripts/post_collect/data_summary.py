import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import tarfile
import time
import sys


def format_id(id_str):
    """
    将ID格式化为5位数，不足5位前面补0

    参数:
    id_str: 输入的ID字符串

    返回:
    格式化后的5位数字符串
    """
    try:
        # 确保输入是数字
        num = int(id_str)
        # 格式化为5位数，不足补0
        return f"{num:05d}"
    except ValueError:
        raise ValueError(f"ID必须是数字，当前输入: {id_str}")


def load_verb_info():
    """
    加载cand_robot_verb_info.json文件

    返回:
    包含所有动词信息的字典
    """
    try:
        with open('cand_robot_verb_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("找不到cand_robot_verb_info.json文件")
    except json.JSONDecodeError:
        raise ValueError("cand_robot_verb_info.json文件格式错误")


def find_verb_info(verb_info_dict, node, verb):
    """
    在verb_info_dict中查找匹配的node和verb信息

    参数:
    verb_info_dict: 动词信息字典
    node: 节点名称
    verb: 动词

    返回:
    匹配的动词信息字典，如果没有找到则返回None
    """
    for info in verb_info_dict.values():
        if info['node'] == node and info['verb'] == verb:
            return info
    return None


def get_user_nodes():
    """
    获取用户输入的node和verb信息

    返回:
    包含所有node信息的列表
    """
    nodes = []
    verb_info_dict = load_verb_info()

    while True:
        print("\n请输入node和verb信息（输入'q'结束）：")
        node = input("请输入node (例如: hold-15.1-1): ").strip()

        if node.lower() == 'q':
            break

        verb = input("请输入verb (例如: grasp): ").strip()

        if verb.lower() == 'q':
            break

        # 查找匹配的信息
        verb_info = find_verb_info(verb_info_dict, node, verb)
        if verb_info:
            nodes.append(verb_info)
            print(f"已找到匹配信息：{node} - {verb}")
        else:
            print(f"警告：未找到匹配的node和verb组合：{node} - {verb}")
            continue

        # 询问是否继续
        if input("\n是否继续输入？(y/n): ").strip().lower() != 'y':
            break

    return nodes


def get_folder_size(folder_path):
    """
    获取文件夹大小（以字节为单位）
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def format_size(size_bytes):
    """
    将字节大小转换为人类可读的格式
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_tar_gz_with_progress(source_path, output_path):
    """
    创建tar.gz压缩包并显示进度

    参数:
    source_path: 源文件夹路径
    output_path: 输出压缩包路径
    """
    total_size = get_folder_size(source_path)
    processed_size = 0
    start_time = time.time()

    print(f"\n开始压缩文件夹: {source_path}")
    print(f"总大小: {format_size(total_size)}")

    def progress_callback(tarinfo):
        nonlocal processed_size
        processed_size += tarinfo.size
        elapsed_time = time.time() - start_time
        speed = processed_size / elapsed_time if elapsed_time > 0 else 0
        progress = (processed_size / total_size) * 100 if total_size > 0 else 0

        # 计算预计剩余时间
        remaining_size = total_size - processed_size
        eta = remaining_size / speed if speed > 0 else 0

        sys.stdout.write(f"\r压缩进度: {progress:.1f}% | "
                         f"已处理: {format_size(processed_size)} | "
                         f"速度: {format_size(speed)}/s | "
                         f"预计剩余时间: {eta/60:.1f}分钟")
        sys.stdout.flush()
        return tarinfo

    try:
        with tarfile.open(output_path, "w:gz", compresslevel=6) as tar:
            tar.add(source_path, arcname=os.path.basename(source_path),
                    filter=progress_callback)
        print("\n压缩完成！")
    except Exception as e:
        print(f"\n压缩过程中出错: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def process_folder(folder_path, task_id, user_id, scene_id, task_description_english):
    """
    处理指定文件夹，创建metadata.json并重命名文件夹，最后压缩成tar.gz

    参数:
    folder_path: 原始文件夹路径
    task_id: 任务ID
    user_id: 用户ID
    scene_id: 场景ID
    task_description_english: 英文任务描述
    """
    # 确保输入路径存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹 {folder_path} 不存在")

    # 格式化所有ID
    task_id = format_id(task_id)
    user_id = format_id(user_id)
    scene_id = format_id(scene_id)

    # 获取所有hdf5文件
    hdf5_files = [f for f in os.listdir(folder_path) if f.startswith(
        'episode_') and f.endswith('.hdf5')]

    if not hdf5_files:
        raise ValueError(f"在 {folder_path} 中没有找到episode_*.hdf5文件")

    # 获取用户输入的nodes信息
    nodes = get_user_nodes()

    # 创建metadata.json
    metadata = {
        "task_description_english": task_description_english,
        "nodes": nodes
    }

    # 写入metadata.json
    metadata_path = os.path.join(folder_path, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    # 创建新的文件夹名称
    new_folder_name = f"task_{task_id}_user_{user_id}_scene_{scene_id}"
    new_folder_path = new_folder_name

    # 重命名文件夹
    if os.path.exists(new_folder_path):
        raise FileExistsError(f"目标文件夹 {new_folder_path} 已存在")

    shutil.move(folder_path, new_folder_path)

    # 输出metadata.json内容
    print("\nmetadata.json 内容:")
    print("-" * 50)
    print(json.dumps(metadata, indent=4, ensure_ascii=False))
    print("-" * 50)

    # 创建tar.gz压缩包
    tar_gz_path = f"{new_folder_path}.tar.gz"
    create_tar_gz_with_progress(new_folder_path, tar_gz_path)

    print(f"\n处理完成！")
    print(f"新文件夹路径: {new_folder_path}")
    print(f"已创建metadata.json文件")
    print(f"任务描述: {task_description_english}")
    print(f"已添加 {len(nodes)} 个node信息")
    print(f"已创建压缩包: {tar_gz_path}")


def main():
    # 获取用户输入
    folder_path = input("请输入要处理的文件夹路径: ").strip()

    while True:
        try:
            task_id = input("请输入task_id: ").strip()
            task_id = format_id(task_id)
            break
        except ValueError as e:
            print(f"错误: {str(e)}")

    while True:
        try:
            user_id = input("请输入user_id: ").strip()
            user_id = format_id(user_id)
            break
        except ValueError as e:
            print(f"错误: {str(e)}")

    while True:
        try:
            scene_id = input("请输入scene_id: ").strip()
            scene_id = format_id(scene_id)
            break
        except ValueError as e:
            print(f"错误: {str(e)}")

    # 获取任务描述
    task_description_english = input("请输入英文任务描述: ").strip()

    try:
        process_folder(folder_path, task_id, user_id,
                       scene_id, task_description_english)
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
