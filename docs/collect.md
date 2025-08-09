# Collect scripts

## 0. 环境准备
在进行数据采集前，请确保机器已经运行过初始化脚本。
```bash
# 初始化脚本
sh scripts/init.sh
```
插入两个主臂（即遥操作臂）的航插线，重启机械臂插排；运行以下命令进入 collect 模式。
```bash
conda activate aloha
# 如果使用的是 zsh，可以使用 init_collect 快速切换 collect 模式。
init_collect
# 或者手动运行以下脚本
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false 
```

## 1. 采集数据
建议使用以下脚本采集数据集
```bash
# 开始采集数据，操作主臂，--dataset_dir 存储路径，--max_timesteps采集的action数，默认相机频率为25hz, 即25步为1s，--episode_num 轨迹数量，--task_name 任务名称，记得更改防止覆盖其他人数据
python  -m scripts.collect.collect_dataset_eef_qpos.py --dataset_dir=./data --task_name task_test --max_timesteps 1200 --episode_num 42 
```
该脚本会在指定的目录下生成 episode_{idx}.hdf5 文件，
每个文件包含一个 episode 的数据，包括视频、关节角度、末端执行器位置等信息。
请注意，采集过程中需要手动操作主臂进行任务执行。