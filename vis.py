import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch.utils.data import DataLoader

from sign_dataset import Sign_Dataset


subset = "asl100"

split_file = "./data/splits/{}.json".format(subset)

dataset = Sign_Dataset(
    index_file_path=split_file,
    split="train",
    pose_root="./data/pose_per_individual_videos",
    # sample_strategy="rnd_start",
    num_samples=100,
    num_copies=0,
    img_transforms=None,
    video_transforms=None,
    include_confidence=True,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

x, y, video_id = next(iter(dataloader))

num_frames = x.shape[2] // 3
poses = x[0].reshape(55, num_frames, 3).numpy()

fig, ax = plt.subplots(figsize=(10, 10))

scatter = ax.scatter([], [], c="r", s=50)
lines = [ax.plot([], [], c="b")[0] for _ in range(55)]

ax.set_xlim(poses[:, :, 0].min(), poses[:, :, 0].max())
ax.set_ylim(poses[:, :, 1].min(), poses[:, :, 1].max())
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title(f"Pose Animation from video: {video_id[0]}")

connections = [
    (1, 2, "R_Shoulder"),
    (2, 3, "R_Elbow"),
    (3, 4, "R_Wrist"),
    (1, 5, "L_Shoulder"),
    (5, 6, "L_Elbow"),
    (6, 7, "L_Wrist"),
    (1, 8, "Mid_Hip"),
    (9, 10, "Eyes"),
    (9, 11, "R_Ear"),
    (12, 10, "L_Ear"),
    # Right Hand
    (13, 14, "R_Thumb_IP"),
    (14, 15, "R_Thumb_Tip"),
    (4, 16, "R_Index_MCP"),
    (16, 17, "R_Index_PIP"),
    (17, 18, "R_Index_DIP"),
    (18, 19, "R_Index_Tip"),
    (4, 20, "R_Middle_MCP"),
    (20, 21, "R_Middle_PIP"),
    (21, 22, "R_Middle_DIP"),
    (22, 23, "R_Middle_Tip"),
    (4, 24, "R_Ring_MCP"),
    (24, 25, "R_Ring_PIP"),
    (25, 26, "R_Ring_DIP"),
    (26, 27, "R_Ring_Tip"),
    (4, 28, "R_Pinky_MCP"),
    (28, 29, "R_Pinky_PIP"),
    (29, 30, "R_Pinky_DIP"),
    (30, 31, "R_Pinky_Tip"),
    # Left Hand
    (7, 32, "L_Thumb_CMC"),
    (32, 33, "L_Thumb_MCP"),
    (33, 34, "L_Thumb_IP"),
    (34, 35, "L_Thumb_Tip"),
    (7, 36, "L_Index_MCP"),
    (36, 37, "L_Index_PIP"),
    (37, 38, "L_Index_DIP"),
    (38, 39, "L_Index_Tip"),
    (7, 40, "L_Middle_MCP"),
    (40, 41, "L_Middle_PIP"),
    (41, 42, "L_Middle_DIP"),
    (42, 43, "L_Middle_Tip"),
    (7, 44, "L_Ring_MCP"),
    (44, 45, "L_Ring_PIP"),
    (45, 46, "L_Ring_DIP"),
    (46, 47, "L_Ring_Tip"),
    (7, 48, "L_Pinky_MCP"),
    (48, 49, "L_Pinky_PIP"),
    (49, 50, "L_Pinky_DIP"),
    (50, 51, "L_Pinky_Tip"),
]

labels = [ax.text(0, 0, "", fontsize=8, color="green") for _ in connections]

CONFIDENCE_THRESHOLD = 0.001
def update(frame):
    frame_data = poses[:, frame, :]
    confident_points = frame_data[:, 2] > CONFIDENCE_THRESHOLD

    scatter.set_offsets(frame_data[confident_points, :2])

    for i, (start, end, label_text) in enumerate(connections):
        if confident_points[start] and confident_points[end]:
            start_point = frame_data[start, :2]
            end_point = frame_data[end, :2]
            lines[i].set_data(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
            )
            # Update label position to the midpoint of the connection
            label_pos = ((start_point + end_point) / 2)
            labels[i].set_position(label_pos)
            labels[i].set_text(label_text)
        else:
            lines[i].set_data([], [])
            labels[i].set_text("")

    return scatter, *lines, *labels

anim = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=1000 / 30, blit=True, repeat=True
)

# Add pause/resume functionality
paused = False
def on_key_press(event):
    global paused
    if event.key == ' ':
        if paused:
            anim.resume()
        else:
            anim.pause()
        paused = not paused

fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()