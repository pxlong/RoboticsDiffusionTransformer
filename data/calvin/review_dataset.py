import os
import h5py
from collections import Counter
import matplotlib.pyplot as plt

# Define the folder path
folder_path = "/home/longpinxin/ws/data/calvin/task_D_D/new/training"

# Initialize a Counter object to count the frequency of instructions
task_counter = Counter()
task_instruction = {}
# instruction_counter = Counter()

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".hdf5"):  # Process only .hdf5 files
        file_path = os.path.join(folder_path, filename)

        # Open the HDF5 file
        with h5py.File(file_path, "r") as f:
            # Check if the "instruction" attribute exists
            if "instruction" in f.attrs and "task" in f.attrs:
                task = f.attrs["task"]
                task_counter[task] += 1

                instruction = f.attrs["instruction"]
                if task not in task_instruction:
                    task_instruction[task] = []
                if instruction not in task_instruction[task]:
                    task_instruction[task].append(instruction)

# Save the statistics to a .txt file
output_file_path = "results.txt"
with open(output_file_path, "w") as file:
    for task, count in task_counter.items():
        file.write(f"task: {task}, count: {count}, ")
        file.write(f"instruction: {task_instruction[task]}\n")
        file.write("\n")

# Print the statistics
for task, count in task_counter.items():
    print(f"task: {task}, Count: {count}, ")
    print(f"instruction: {task_instruction[task]}")

# Prepare data for plotting
keys = list(task_counter.keys())
values = list(task_counter.values())

# Plot the statistics
plt.figure(figsize=(10, 6))
plt.bar(keys, values, color="skyblue")
plt.xlabel("Tasks")
plt.ylabel("Count")
plt.title("Frequency of Tasks in HDF5 Files")
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Save the plot as an image
output_image_path = "task_frequency.png"
plt.savefig(output_image_path)

# Show the plot (optional)
plt.show()

print(f"Plot saved as: {output_image_path}")
