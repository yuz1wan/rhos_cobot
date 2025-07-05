# Post Collection Scripts

采集完数据后，使用以下脚本进行验证以及数据处理。

## 0. 前后版本差异问题
由于采集脚本的版本变化问题，导致部分采集到的数据并没有正确标注 compress 标志位。
在运行以下脚本前，请

## 1. 可视化数据
首先，请可视化采集到的视频、关节角度、末端执行器等数据。  
For example:

```bash
python -m rhos_cobot.post_collect.visualize_episodes_eef.py --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5
```
该脚本会在指定的目录下生成可视化结果，包括视频、关节角度图像等。请仔细观察是否存在视频数据损坏、关节角度记录异常（如左右臂数据明显颠倒）、末端执行器数据异常等问题。

## 2. 重播数据（可选）
在可视化检查通过后，可以使用以下脚本重播数据，验证数据的完整性和正确性。  
⚠️Warning⚠️  运行重播前请确保：  
重播数据需要确保采集时的环境和设备状态与实际使用时一致;
机器处于 deploy模式，即可以接收关节角度和末端执行器位置的指令，详见 [deploy模式](./deploy.md)。

```bash
python -m rhos_cobot.post_collect.replay_data_eef.py --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5

python -m rhos_cobot.post_collect.replay_data_joint.py --dataset_dir ./data/ --task_name task0063_user0012_scene0004_ep0 --episode_idx 5
```
该脚本会重播指定的episode数据，并在控制台输出关节角度和末端执行器位置等信息。

## 3. 数据处理
如果数据可视化和重播都通过了，可以使用以下脚本进行数据处理，对数据命名以及元数据进行格式统一。

### 3.1 计算时长
```bash
# 计算该任务总时长，填入飞书表格，相机频率默认为25hz，即25步为1s，修改相机频率后请更改 --camera_fps
python -m rhos_cobot.post_collect.cal_time.py --dataset_dir ./data/task0063_user0012_scene0004_ep0 --camera_fps 25
```
### 3.2 规范化数据名
统一数据命名为**task_{task_id}_user_{user_id}_scene_{scene_id}**  
task_id：对应飞书表格前面序号  
user_id：对应飞书表格分配的序号  
scene_id：对应同一个任务不同场景的数据（一个任务可能在不同任务场景，或者改变一些变量多次采集）  

```bash
# 规范化数据名
python -m rhos_cobot.post_collect.data_summary_simple.py
```
要求输入
1. 处理的文件夹路径，即刚刚采集完的所有episode_{idx}.hdf5文件所在路径
2. task_id， user_id， scene_id

### 3.3 生成json存储元数据

## 4. 统一上传