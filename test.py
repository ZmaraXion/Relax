import torch
import cv2
import numpy as np
import pyiqa
import matplotlib.pyplot as plt

class VideoAnomalyDetector:
    def __init__(self, metric_name='maniqa', device='cuda'):
        """
        初始化偵測器
        model_name: 推薦 'maniqa' (高精確度) 或 'musiq' (Google開發, 魯棒性佳)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading IQA Model: {metric_name} on {self.device}...")
        # PyIQA 自動下載預訓練權重
        self.iqa_metric = pyiqa.create_metric(metric_name, device=self.device)
        
    def get_scene_diff(self, frame1, frame2):
        """
        計算簡單的結構/色彩差異，用於判斷是否為 Scene Cut
        使用 Histogram Intersection
        """
        h1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h1, h1).flatten()
        cv2.normalize(h2, h2).flatten()
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    def detect(self, video_path, scene_cut_thresh=0.85, z_score_thresh=3.0):
        cap = cv2.VideoCapture(video_path)
        scores = []
        scene_diffs = []
        frame_indices = []
        
        prev_frame = None
        idx = 0
        
        print("Starting analysis...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 轉換為 Tensor 並計算 IQA 分數
            # PyIQA 接受 0-1 的 RGB Tensor (NCHW)
            img_tensor = pyiqa.utils.img2tensor(frame).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                # score 通常越高代表品質越好
                score = self.iqa_metric(img_tensor).item()
            
            # 2. 計算與上一幀的內容相似度 (Scene Cut Detection)
            is_scene_cut = False
            if prev_frame is not None:
                content_sim = self.get_scene_diff(prev_frame, frame)
                if content_sim < scene_cut_thresh:
                    is_scene_cut = True
                    # 轉場時，強制將品質變化視為無效或標記特殊
            else:
                content_sim = 1.0 # 第一幀
            
            scores.append(score)
            scene_diffs.append(content_sim)
            frame_indices.append(idx)
            
            prev_frame = frame
            idx += 1
            
            if idx % 100 == 0:
                print(f"Processed {idx} frames...")

        cap.release()
        
        # 3. 後處理：偵測異常跳變
        scores = np.array(scores)
        # 計算分數的一階差分 (Delta)
        deltas = np.diff(scores, prepend=scores[0])
        
        # 使用滑動窗口計算局部 Mean 和 Std 進行 Z-score 異常偵測
        window_size = 30
        anomalies = []
        
        for i in range(len(deltas)):
            # 忽略開頭與 Scene Cut
            if scene_diffs[i] < scene_cut_thresh:
                continue
                
            start = max(0, i - window_size)
            end = min(len(deltas), i + window_size)
            local_mean = np.mean(deltas[start:end])
            local_std = np.std(deltas[start:end]) + 1e-6 # 避免除以0
            
            z_score = abs(deltas[i] - local_mean) / local_std
            
            # 異常定義：分數劇烈下降 (Delta 為負且很大) 且 Z-score 超標
            # 若要偵測變好或變壞，則只看 abs(z_score)
            # 這裡假設我們要抓「變模糊/變差」，所以 deltas[i] < 0
            if z_score > z_score_thresh and abs(deltas[i]) > 0.05: # 0.05 是絕對分數變化的最小閾值，需根據模型範圍調整
                anomalies.append({
                    'frame_idx': i,
                    'score': scores[i],
                    'prev_score': scores[i-1],
                    'delta': deltas[i],
                    'z_score': z_score,
                    'type': 'Quality Drop' if deltas[i] < 0 else 'Quality Spike'
                })
                
        return anomalies, scores

# 使用範例
if __name__ == "__main__":
    # 需要先下載一個測試影片
    detector = VideoAnomalyDetector(metric_name='maniqa') # MANIQA 分數範圍通常 0~1
    anomalies, score_trace = detector.detect('test_video.mp4')
    
    print("\nDetected Anomalies:")
    for a in anomalies:
        print(f"Frame {a['frame_idx']}: Delta {a['delta']:.4f} (Z: {a['z_score']:.2f}) - {a['type']}")
