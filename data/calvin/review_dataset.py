import os
import h5py
from collections import Counter

dataset_dirs = {
    "train": "/mnt/petrelfs/longpinxin/data/calvin/training",
    "validation": "/mnt/petrelfs/longpinxin/data/calvin/validation",
}

folder_path = dataset_dirs["train"]
print(f"dataset path: {folder_path}")

instruction_counter = Counter()

for filename in os.listdir(folder_path):
    if filename.endswith(".hdf5"):  
        file_path = os.path.join(folder_path, filename)
        
        with h5py.File(file_path, 'r') as f:
            if "instruction" in f.attrs:
                instruction = f.attrs["instruction"]
                instruction_counter[instruction] += 1

for instruction, count in instruction_counter.items():
    print(f"Instruction: {instruction}, Count: {count}")