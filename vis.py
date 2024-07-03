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
idk = iter(dataloader)
video_idx = 7
for i in range(video_idx):
    next(idk)
x, y, video_id = next(idk)

num_frames = x.shape[2] // 3
poses = x[0].reshape(55, num_frames, 3).numpy()

fig, ax = plt.subplots(figsize=(10, 10))

scatter = ax.scatter([], [], c="r", s=25)
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
    # Left Hand
    (7, 14, "R_Wrist_to_Thumb"),
    (14, 15, "idk"),
    (15, 16, "idk"),
    (16, 17, "idk"),
    (7, 18, "idk"),
    (18, 19, "idk"),
    (19, 20, "idk"),
    (20, 21, "idk"),
    (7, 22, "idk"),
    (22, 23, "idk"),
    (23, 24, "idk"),
    (24, 25, "idk"),
    (7, 26, "idk"),
    (26, 27, "idk"),
    (27, 28, "idk"),
    (28, 29, "idk"),
    (7, 30, "idk"),
    (30, 31, "idk"),
    (31, 32, "idk"),
    (32, 33, "idk"),
    # Right Hand
    (4, 34, "L_Wrist_to_Thumb"),
    (34, 35, "idk"),
    (35, 36, "idk"),    
    (36, 37, "idk"),
    (37, 38, "idk"),
    (4, 39, "idk"),
    (39, 40, "idk"),
    (40, 41, "idk"),
    (41, 42, "idk"),
    (4, 43, "idk"),
    (43, 44, "idk"),
    (44, 45, "idk"),
    (45, 46, "idk"),
    (4, 47, "idk"),
    (47, 48, "idk"),
    (48, 49, "idk"),
    (49, 50, "idk"),
    (4, 51, "idk"),
    (51, 52, "idk"),
    (52, 53, "idk"),
    (53, 54, "idk"),
]

labels = [ax.text(0, 0, "", fontsize=8, color="green") for _ in connections]

CONFIDENCE_THRESHOLD = 0.0

current_frame = 0
anim = None

def update(frame_num):
    global current_frame
    current_frame = frame_num
    frame_data = poses[:, frame_num, :]
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

paused = False
def on_key_press(event):
    global paused, current_frame, anim
    if event.key == ' ':
        if paused:
            anim.resume()
        else:
            anim.pause()
        paused = not paused
    elif event.key == 'right' and paused:
        current_frame = (current_frame + 1) % num_frames
        anim.frame_seq = anim.new_frame_seq()
        [next(anim.frame_seq) for _ in range(current_frame)]
        anim._draw_next_frame(None, blit=True)
    elif event.key == 'left' and paused:
        current_frame = (current_frame - 1) % num_frames
        anim.frame_seq = anim.new_frame_seq()
        [next(anim.frame_seq) for _ in range(current_frame)]
        anim._draw_next_frame(None, blit=True)

fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()