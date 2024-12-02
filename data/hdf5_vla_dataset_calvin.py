import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

# from data.rotation_conversions import euler_angles_to_matrix, matrix_to_rotation_6d
from data.test_6drot import (
    convert_euler_to_rotation_matrix,
    compute_ortho6d_from_rotation_matrix,
)
from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """

    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        # HDF5_DIR = "data/datasets/agilex/rdt_data/"
        # HDF5_DIR = "/home/longpinxin/ws/data/calvin/task_D_D/test/"
        HDF5_DIR = "/mnt/petrelfs/longpinxin/data/calvin/training"
        self.DATASET_NAME = "calvin_d_d"

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        # Load the config
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]
        # print(
        #     f"CHUNK_SIZE: {self.CHUNK_SIZE}, IMG_HISORY_SIZE: {self.IMG_HISORY_SIZE}, STATE_DIM: {self.STATE_DIM}"
        # )

        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res["state"].shape[0] if valid else 0
            # print(f"Episode {file_path} has length {_len}")
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def convert_action(self, input):
        eef_pos = input[:, :3]
        eef_ang = convert_euler_to_rotation_matrix(input[:, 3:6])
        eef_ang = compute_ortho6d_from_rotation_matrix(eef_ang)
        gripper_open = (input[:, 6] + 1) / 2
        gripper_open = gripper_open[:, np.newaxis]
        # print(
        #     f"eef_pos shape: {eef_pos.shape}, eef_ang shape: {eef_ang.shape}, gripper_open shape: {gripper_open.shape}"
        # )
        output = np.concatenate([gripper_open, eef_pos, eef_ang], axis=1)
        return output

    def convert_robot_obs(self, input):
        eef_pos = input[:, :3]
        eef_ang = convert_euler_to_rotation_matrix(input[:, 3:6])
        eef_ang = compute_ortho6d_from_rotation_matrix(eef_ang)
        gripper_open = (input[:, 14] + 1) / 2
        gripper_open = gripper_open[:, np.newaxis]
        # qpos = input[:, 7:14]
        # print(
        #     f"eef_pos shape: {eef_pos.shape}, eef_ang shape: {eef_ang.shape}, gripper_open shape: {gripper_open.shape}, qpos shape: {qpos.shape}"
        # )
        output = np.concatenate([gripper_open, eef_pos, eef_ang], axis=1)
        return output

    def clean_task_instruction(self, task_instruction: str, replacements: dict) -> str:
        """
        Clean up the natural language task instruction.
        """
        # Apply replacements to the string
        for old, new in replacements.items():
            task_instruction = task_instruction.replace(old, new)

        # Strip leading and trailing spaces
        cleaned_task_instruction = task_instruction.strip()

        return cleaned_task_instruction

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(
                    self.file_paths, p=self.episode_sample_weights
                )
            else:
                file_path = self.file_paths[index]
            valid, sample = (
                self.parse_hdf5_file(file_path)
                if not state_only
                else self.parse_hdf5_file_state_only(file_path)
            )
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, "r") as f:
            robot_obs = f["robot_obs"][()]
            num_steps = robot_obs.shape[0]
            # print(f"num_steps: {num_steps}")
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            #     return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(robot_obs - robot_obs[0:1])
            # print(f"qpos_delta: {qpos_delta}")
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            # print(f"first_idx: {first_idx}")

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx - 1, num_steps)
            # print(f"step_id: {step_id}")

            # Load the instruction
            # dir_path = os.path.dirname(file_path)
            # with open(
            #     os.path.join(dir_path, "expanded_instruction_gpt-4-turbo.json"), "r"
            # ) as f_instr:
            #     instruction_dict = json.load(f_instr)
            # # We have 1/3 prob to use original instruction,
            # # 1/3 to use simplified instruction,
            # # and 1/3 to use expanded instruction.
            # instruction_type = np.random.choice(
            #     ["instruction", "simplified_instruction", "expanded_instruction"]
            # )
            # instruction = instruction_dict[instruction_type]
            # if isinstance(instruction, list):
            #     instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            # instruction = "path/to/lang_embed.pt"
            instruction = f.attrs["instruction"]
            # print(f"instruction before: {instruction}")
            replacements = {
                "_": " ",
                "1f": " ",
                "4f": " ",
                "-": " ",
                "50": " ",
                "55": " ",
                "56": " ",
            }
            instruction = self.clean_task_instruction(instruction, replacements)
            # print(f"instruction after: {instruction}")

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
            }
            # print(f"meta_data: {meta}")

            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #     [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]
            # )
            # target_qpos = f["action"][step_id : step_id + self.CHUNK_SIZE] / np.array(
            #     [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            # )

            # Parse the state and action

            # print(f"state before: {robot_obs[step_id : step_id + 1]}")
            robot_obs = self.convert_robot_obs(robot_obs)
            state = robot_obs[step_id : step_id + 1]

            state_std = np.std(robot_obs, axis=0)
            state_mean = np.mean(robot_obs, axis=0)
            state_norm = np.sqrt(np.mean(robot_obs**2, axis=0))
            # print(f"state after: {state}, state_std: {state_std}, state_mean: {state_mean}, state_norm: {state_norm}")

            actions = f["action"][()]
            actions = self.convert_action(actions)

            actions = actions[step_id : step_id + self.CHUNK_SIZE]
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate(
                    [
                        actions,
                        np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                    ],
                    axis=0,
                )
            # print(f"actions shape: {actions.shape}")

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # print(f"state[0]: {values[0]}")
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = (
                    # [STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)]
                    [STATE_VEC_IDX_MAPPING["gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_x"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_y"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_z"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_0"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_1"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_2"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_3"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_4"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_5"]]
                )
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                # print(f"uni_vec before: {uni_vec[0]}")
                uni_vec[..., UNI_STATE_INDICES] = values
                # print(f"uni_vec after: {uni_vec[0]}")
                return uni_vec

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            # print(f"actions[10]: {actions[10]}, shape: {actions.shape}")

            # Parse the images
            def parse_img(key):
                # print(f"key: {key}")
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    # print(f"img idx: {i}")
                    img = f[key][i]
                    # print(f"img shape: {img.shape}")
                    imgs.append(
                        cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    )
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate(
                        [
                            np.tile(
                                imgs[:1],
                                (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1),
                            ),
                            imgs,
                        ],
                        axis=0,
                    )
                # print(f"imgs shape: {imgs.shape}")
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img("rgb_static")
            # print(f"cam_high shape: {cam_high.shape}")
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            # print(f"valid_len: {valid_len}")
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            # print(f"cam_high_mask: {cam_high_mask}, shape: {cam_high_mask.shape}")

            cam_right_wrist = parse_img("rgb_gripper")
            # print(f"cam_right_wrist shape: {cam_right_wrist.shape}")
            cam_right_wrist_mask = cam_high_mask.copy()
            # print(f"cam_right_wrist_mask: {cam_right_wrist_mask}, shape: {cam_right_wrist.shape}")
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            # print(f"cam_left_wrist shape: {cam_left_wrist.shape}")
            # print(f"cam_left_wrist_mask shape: {cam_left_wrist_mask.shape}")

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, "r") as f:
            robot_obs = f["robot_obs"][()]
            actions = f["action"][()]
            # qpos = f["observations"]["qpos"][:]
            num_steps = robot_obs.shape[0]
            # print(f"num_steps: {num_steps}")
            # [Optional] We drop too-short episode
            # if num_steps < 128:
            #     return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(robot_obs - robot_obs[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            # print(f"first_idx: {first_idx}")

            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #     [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]]
            # )
            # target_qpos = f["action"][:] / np.array(
            #     [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            # )

            # Parse the state and action
            state = robot_obs[first_idx - 1 :]
            action = actions[first_idx - 1 :]

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                # print(f"state[11]: {values[11]}")
                UNI_STATE_INDICES = (
                    # [STATE_VEC_IDX_MAPPING[f"arm_joint_{i}_pos"] for i in range(7)]
                    [STATE_VEC_IDX_MAPPING["gripper_open"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_x"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_y"]]
                    + [STATE_VEC_IDX_MAPPING["eef_pos_z"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_0"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_1"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_2"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_3"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_4"]]
                    + [STATE_VEC_IDX_MAPPING["eef_angle_5"]]
                )
                # print(f"UNI_STATE_INDICES: {UNI_STATE_INDICES}")
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                # print(f"uni_vec before: {uni_vec[11]}")
                uni_vec[..., UNI_STATE_INDICES] = values
                # print(f"uni_vec after: {uni_vec[11]}")
                return uni_vec

            # print(f"state shape {state.shape} before convert: {state[0, :]}")
            state = self.convert_robot_obs(state)
            # print(f"state shape {state.shape} after convert: {state[0, :]}")
            state = fill_in_state(state)

            # print(f"action before convert: {action[0, :]}")
            action = self.convert_action(action)
            # print(f"action after convert: {action[0, :]}")
            action = fill_in_state(action)

            # Return the resulting sample
            return True, {"state": state, "action": action}


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)