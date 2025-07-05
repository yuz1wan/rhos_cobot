import os
import h5py
import argparse

def compute_total_qpos_length_and_duration(dataset_dir, camera_fps=25):
    """
    Compute the total number of timesteps from the '/observations/qpos' dataset
    in all .hdf5 files under the specified directory. Then calculate the total duration
    based on the given camera frame rate.

    Args:
        dataset_dir (str): Path to the directory containing .hdf5 files.
        camera_fps (float): Frame rate of the camera (default: 25 Hz).
    """
    total_steps = 0  # Accumulate total number of qpos steps across all files

    # List all .hdf5 files in the given directory
    hdf5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])

    if not hdf5_files:
        print("No .hdf5 files found in the directory.")
        return

    # Iterate through each hdf5 file
    for filename in hdf5_files:
        file_path = os.path.join(dataset_dir, filename)
        try:
            # Open the HDF5 file in read mode
            with h5py.File(file_path, 'r') as f:
                # Access the qpos dataset to get the number of timesteps (first dimension)
                qpos = f['/observations/qpos']
                steps = qpos.shape[0]
                print(f'{filename}: {steps} steps')
                total_steps += steps
        except Exception as e:
            # Handle any errors encountered while reading the file
            print(f"Failed to read {filename}: {e}")

    # Compute total duration in seconds
    total_seconds = total_steps / camera_fps

    # Print summary
    print(f'\n=== Summary ===')
    print(f'Total files: {len(hdf5_files)}')
    print(f'Total steps: {total_steps}')
    print(f'Total duration: {total_seconds:.2f} seconds ({total_seconds / 60:.2f} minutes)')

# Entry point when run as a script
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Compute total qpos steps and duration from HDF5 dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing .hdf5 files.")
    parser.add_argument("--camera_fps", type=float, default=25.0,
                        help="Camera frequency in Hz (default: 25)")

    # Parse arguments and run computation
    args = parser.parse_args()
    compute_total_qpos_length_and_duration(args.dataset_dir, args.camera_fps)

# python cal_time.py --dataset_dir your/path --camera_fps 25