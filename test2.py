import cv2
import numpy as np
import random

# 參數設置
video1_path = "video1.mp4"
video2_path = "video2.mp4"
output_dir = "output/"
height, width = 480, 640
group_size = 5

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

# 規範化軌跡
def normalize_tracks(tracks, height, width):
    normalized = []
    for track in tracks:
        norm_track = [(pt_idx, [pt[0] / width, pt[1] / height]) for pt_idx, pt in track]
        normalized.append(norm_track)
    return normalized

def filter_tracks(tracks, min_length=2, max_length=5):
    return [track for track in tracks if min_length <= len(track) <= max_length]

# 計算軌跡距離
def compute_trajectory_distance(tracks1, tracks2, height, width):
    norm_tracks1 = normalize_tracks(tracks1, height, width)
    norm_tracks2 = normalize_tracks(tracks2, height, width)
    
    distances = []
    for t1 in norm_tracks1:
        for t2 in norm_tracks2:
            if len(t1) == len(t2):
                dist = np.mean([np.linalg.norm(np.array(t1[i][1]) - np.array(t2[i][1])) for i in range(len(t1))])
                distances.append(dist)
    return np.mean(distances) if distances else float('inf')

# 幾何一致性檢查
def compute_homography(tracks1, tracks2):
    if len(tracks1) < 4 or len(tracks2) < 4:
        return None, []
    pts1 = np.float32([t[0][1] for t in tracks1]).reshape(-1, 1, 2)
    pts2 = np.float32([t[0][1] for t in tracks2]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = [i for i, m in enumerate(mask) if m]
    return H, inliers

# 匹配 group
def match_groups(tracks_v1, tracks_v2, height, width):
    best_group_id = -1
    min_distance = float('inf')
    best_inliers = 0
    
    for group_id, tracks in tracks_v2.items():
        dist = compute_trajectory_distance(tracks_v1, tracks, height, width)
        H, inliers = compute_homography(tracks_v1, tracks, height, width)
        inlier_count = len(inliers)
        
        score = dist / (inlier_count + 1)
        if score < min_distance:
            min_distance = score
            best_group_id = group_id
            best_inliers = inlier_count
    
    return best_group_id, min_distance, best_inliers

# 選擇隨機 group
def select_random_group(frames, group_size=5):
    start_idx = random.randint(0, len(frames) - group_size)
    return frames[start_idx:start_idx + group_size], start_idx

# 可視化對齊結果
def visualize_alignment(frames_v1, frames_v2, tracks_v1, tracks_v2, start_idx_v1, group_id_v2):
    group_v1 = frames_v1[start_idx_v1:start_idx_v1 + 5]
    group_v2 = frames_v2[group_id_v2:group_id_v2 + 5]
    
    output_frames_v1 = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in group_v1]
    output_frames_v2 = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in group_v2]
    
    for track in tracks_v1:
        for i, (pt_idx, pt) in enumerate(track):
            if i < len(output_frames_v1):
                pt = pt.astype(int)
                cv2.circle(output_frames_v1[i], tuple(pt), 3, (0, 255, 0), -1)
                if i > 0:
                    pt_prev = track[i - 1][1].astype(int)
                    cv2.line(output_frames_v1[i], tuple(pt_prev), tuple(pt), (0, 0, 255), 1)
    
    for track in tracks_v2[group_id_v2]:
        for i, (pt_idx, pt) in enumerate(track):
            if i < len(output_frames_v2):
                pt = pt.astype(int)
                cv2.circle(output_frames_v2[i], tuple(pt), 3, (0, 255, 0), -1)
                if i > 0:
                    pt_prev = track[i - 1][1].astype(int)
                    cv2.line(output_frames_v2[i], tuple(pt_prev), tuple(pt), (0, 0, 255), 1)
    
    return output_frames_v1, output_frames_v2

# 儲存影片
def save_video(frames, output_path, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

# 主程式
frames_v1 = load_video(video1_path)
frames_v2 = load_video(video2_path)
# 假設 tracks_v1 和 tracks_v2 已從之前的程式碼生成
# tracks_v1: List of tracks for the selected group
# tracks_v2: Dict of {group_id: List of tracks}
start_idx_v1, best_group_id, min_distance, inlier_count = align_videos(frames_v1, frames_v2, tracks_v1, tracks_v2, height, width)
output_frames_v1, output_frames_v2 = visualize_alignment(frames_v1, frames_v2, tracks_v1, tracks_v2, start_idx_v1, best_group_id)
save_video(output_frames_v1, f"{output_dir}/video1_group_{start_idx_v1}.mp4")
save_video(output_frames_v2, f"{output_dir}/video2_group_{best_group_id}.mp4")
print(f"Video1 group starting at frame {start_idx_v1} matches Video2 group starting at frame {best_group_id}")
