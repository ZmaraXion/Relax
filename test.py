import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from accelerated_features import XFeat

# 參數設置
video_path = "input_video.mp4"
output_dir = "output/"
group_size = 5
height, width = 480, 640
min_length = 2
max_length = 5
match_threshold = 0.2

# 讀取影片
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    return frames

def preprocess_frame(frame, height=480, width=640):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    return resized

def generate_frame_groups(frames, group_size=5):
    return [frames[i:i + group_size] for i in range(len(frames) - group_size + 1)]

# 初始化 XFeat 和 LightGlue
xfeat = XFeat(model_path="xfeat_model.pth", device="cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
matcher = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")

def process_frame_group(group):
    keypoints_list, descriptors_list = [], []
    for frame in group:
        keypoints, descriptors = xfeat.detectAndDescribe(frame)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    matches_list = []
    for i in range(len(group) - 1):
        inputs = processor([group[i], group[i + 1]], return_tensors="pt")
        with torch.no_grad():
            outputs = matcher(**inputs)
        image_sizes = [(group[i].shape[0], group[i].shape[1]), (group[i + 1].shape[0], group[i + 1].shape[1])]
        matches = processor.post_process_keypoint_matching(outputs, [image_sizes])[0]
        matches_list.append({
            "keypoints0": matches["keypoints0"].numpy(),
            "keypoints1": matches["keypoints1"].numpy(),
            "scores": matches["matching_scores"].numpy()
        })
    return keypoints_list, matches_list

def generate_tracks(keypoints_list, matches_list, min_length=2, max_length=5):
    tracks = []
    track_id = 0
    active_tracks = {}

    for i, matches in enumerate(matches_list):
        keypoints0 = keypoints_list[i]
        keypoints1 = keypoints_list[i + 1]
        match_indices = np.vstack((matches["keypoints0"], matches["keypoints1"])).T
        match_scores = matches["scores"]

        new_active_tracks = {}
        for m, score in zip(match_indices, match_scores):
            if score < match_threshold:
                continue
            p0_idx, p1_idx = m
            p0 = keypoints0[p0_idx]
            p1 = keypoints1[p1_idx]
            track_found = False
            for tid, track in active_tracks.items():
                if track[-1][0] == p0_idx and len(track) < max_length:
                    track.append((p1_idx, p1))
                    new_active_tracks[tid] = track
                    track_found = True
                    break
            if not track_found:
                new_active_tracks[track_id] = [(p0_idx, p0), (p1_idx, p1)]
                track_id += 1
        
        for tid, track in new_active_tracks.items():
            if len(track) >= min_length:
                tracks.append(track)
        active_tracks = new_active_tracks
    
    return tracks

def visualize_tracks(group, tracks):
    output_frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in group]
    for track in tracks:
        for i, (pt_idx, pt) in enumerate(track):
            if i < len(output_frames):
                pt = pt.astype(int)
                cv2.circle(output_frames[i], tuple(pt), 3, (0, 255, 0), -1)
                if i > 0:
                    pt_prev = track[i - 1][1].astype(int)
                    cv2.line(output_frames[i], tuple(pt_prev), tuple(pt), (0, 0, 255), 1)
    return output_frames

def save_video(frames, output_path, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

def save_tracks(tracks, output_path):
    with open(output_path, 'w') as f:
        for i, track in enumerate(tracks):
            f.write(f"Track {i}:\n")
            for pt_idx, pt in track:
                f.write(f"{pt[0]},{pt[1]}\n")

# 主程式
frames = load_video(video_path)
groups = generate_frame_groups(frames, group_size)

for i, group in enumerate(groups):
    keypoints_list, matches_list = process_frame_group(group)
    tracks = generate_tracks(keypoints_list, matches_list, min_length, max_length)
    output_frames = visualize_tracks(group, tracks)
    save_video(output_frames, f"{output_dir}/group_{i}.mp4")
    save_tracks(tracks, f"{output_dir}/group_{i}_tracks.txt")
