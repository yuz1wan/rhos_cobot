import os
import h5py
import argparse
from rhos_cobot.post_process import get_dataset_seconds

# Entry point when run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute total qpos length and duration from HDF5 files.")
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)

    # Parse arguments and run computation
    args = parser.parse_args()

    total_seconds = get_dataset_seconds(
        os.path.join(args.dataset_dir, args.task_name))

    print(f"Total qpos Duration: {total_seconds} seconds")
