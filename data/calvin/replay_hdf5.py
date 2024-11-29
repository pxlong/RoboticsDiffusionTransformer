import h5py
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def save_video(frames, instruction, output_path, fps=15):
    """Saves a list of frames as an mp4 video."""
    if not frames.any():
        print(f"No frames to save for {output_path}")
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        cv2.putText(
            frame,
            instruction,
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Video saved to {output_path}")


def plot_actions(actions, instruction, episode_name, output_path):
    # Create a figure with 3 subplots sharing the same x-axis
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), dpi=300, sharex=True)

    # Extract position, orientation, and gripper action
    position = actions[:, 0:3]
    orientation = actions[:, 3:6]
    gripper = actions[:, 6]

    # Time steps
    time_steps = np.arange(position.shape[0])

    # Plot position
    axs[0].plot(time_steps, position[:, 0], label="X")
    axs[0].plot(time_steps, position[:, 1], label="Y")
    axs[0].plot(time_steps, position[:, 2], label="Z")
    axs[0].set_title("TCP Position Over Time")
    axs[0].set_ylabel("Position (units)")
    axs[0].legend()

    # Plot orientation (convert to degrees)
    orientation_deg = np.degrees(orientation)
    axs[1].plot(time_steps, orientation_deg[:, 0], label="X")
    axs[1].plot(time_steps, orientation_deg[:, 1], label="Y")
    axs[1].plot(time_steps, orientation_deg[:, 2], label="Z")
    axs[1].set_title("TCP Orientation Over Time (Degrees)")
    axs[1].set_ylabel("Orientation (degrees)")
    axs[1].legend()

    # Plot gripper action as step plot
    axs[2].step(time_steps, gripper, where="post")
    axs[2].set_title("Gripper Action Over Time")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Action")
    axs[2].set_yticks([-1, 1])
    axs[2].set_yticklabels(["Close", "Open"])
    axs[2].grid(True)

    # Add main title
    fig.suptitle(
        f"Actions for Episode {episode_name} - {instruction}",
        fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for suptitle

    # Save the figure
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    home_dir = os.path.expanduser("~")
    train_dir = os.path.join(home_dir, "ws/data/calvin/task_D_D/training")
    valid_dir = os.path.join(home_dir, "ws/data/calvin/task_D_D/validation")
    vis_dir = os.path.join(home_dir, "ws/data/calvin/task_D_D/visualization")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print(
        f"home_dir: {home_dir}, train_dir: {train_dir}, valid_dir: {valid_dir}, vis_dir: {vis_dir}"
    )

    os.chdir(train_dir)
    files = glob.glob("*.hdf5")
    i = 0
    for file in files:
        print()
        print(f"visulizing {file}")
        episode_name = os.path.splitext(os.path.basename(file))[0]
        try:
            with h5py.File(file, "r") as f:
                rgb_static = f["rgb_static"][()]
                actions = f["action"][()]
                instr = f.attrs["instruction"]

            print(f"instruction: {instr}")
            print(f"rgb_static shape: {rgb_static.shape}")
            print(f"actions shape: {actions.shape}")

            # Ensure rgb_static is in uint8 format
            if not np.any(rgb_static):
                rgb_static = (rgb_static * 255).astype(np.uint8)
            # Save video
            video_output = os.path.join(vis_dir, f"{episode_name}.mp4")
            save_video(rgb_static, instr, video_output)
            # Plot actions
            plot_output = os.path.join(vis_dir, f"{episode_name}.png")
            plot_actions(actions, instr, episode_name, plot_output)
        except Exception as e:
            print(f"Error processing {file}: {e}")

        if i == 5:
            break
        i += 1


if __name__ == "__main__":
    main()
