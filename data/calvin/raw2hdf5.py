import os
import numpy as np
from tqdm import tqdm
import h5py


def write_hdf5(root_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    lang_ann_path = os.path.join(root_dir, "lang_annotations/auto_lang_ann.npy")
    if not os.path.exists(lang_ann_path):
        print(
            f"Language annotation file {lang_ann_path} does not exist, skipping this directory."
        )
        return

    f = np.load(lang_ann_path, allow_pickle=True)
    lang = f.item()["language"]["ann"]
    lang_start_end_idx = f.item()["info"]["indx"]
    num_ep = len(lang_start_end_idx)

    with tqdm(total=num_ep) as pbar:
        for episode_idx, (start_idx, end_idx) in enumerate(lang_start_end_idx):
            print(
                f"Processing episode {episode_idx}, start_idx: {start_idx}, end_idx: {end_idx}, episdoe_len: {end_idx - start_idx + 1}"
            )
            pbar.update(1)

            step_files = [
                f"episode_{str(i).zfill(7)}.npz" for i in range(start_idx, end_idx + 1)
            ]
            action = []
            robot_obs = []
            rgb_static = []
            rgb_gripper = []
            instr = lang[episode_idx]
            print(f"Instruction: {instr}")

            for step_file in step_files:
                filepath = os.path.join(root_dir, step_file)
                if not os.path.exists(filepath):
                    print(f"File {filepath} does not exist, skipping this step")
                    continue
                f_step = np.load(filepath)
                action.append(f_step["actions"])
                robot_obs.append(f_step["robot_obs"])
                rgb_static.append(f_step["rgb_static"])
                rgb_gripper.append(f_step["rgb_gripper"])

            hdf5_path = os.path.join(out_dir, f"{episode_idx:07d}.hdf5")
            print(f"Writing HDF5 to {hdf5_path}")
            with h5py.File(hdf5_path, "w") as hf:
                hf.create_dataset("action", data=action)
                hf.create_dataset("robot_obs", data=robot_obs)
                hf.create_dataset("rgb_static", data=rgb_static)
                hf.create_dataset("rgb_gripper", data=rgb_gripper)
                hf.attrs["instruction"] = instr
                hf.attrs["terminate_episode"] = len(action) - 1


home_dir = os.path.expanduser("~")
print(home_dir)
output_dirs = [
    os.path.join(home_dir, "ws/data/calvin/task_D_D/training"),
    os.path.join(home_dir, "ws/data/calvin/task_D_D/validation"),
]

root_dirs = [
    "/media/longpinxin/DATA/px/calvin/data/task_D_D/training",
    "/media/longpinxin/DATA/px/calvin/data/task_D_D/validation",
]

for root_dir, output_dir in zip(root_dirs, output_dirs):
    print(f"loading from {root_dir}, and saving HDF5 to {output_dir}")
    write_hdf5(root_dir, output_dir)
