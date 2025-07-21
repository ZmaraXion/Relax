import cv2
import numpy as np
from tqdm import tqdm
import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

# ==============================================================================
# 方案一：強化局部特徵匹配 (高精度導向)
# ==============================================================================
class EnhancedLocalFeatureMatcher:
    """
    使用強化局部特徵匹配方法，透過全局幾何一致性來同步影片。
    核心策略：
    1. 計算基於聚合特徵的全局 Homography，代表兩個視角的穩定幾何關係。
    2. 使用全局 Homography 引導匹配，將 Video1 的幀校正到 Video2 的視角下。
    3. 使用歸一化互相關 (NCC) 進行精確評分，該指標對光線變化不敏感。
    """

    def __init__(self, n_features: int = 5000, ransac_thresh: float = 5.0):
        """
        初始化 SIFT 偵測器和 FLANN 匹配器。

        Args:
            n_features (int): SIFT 提取的最大特徵點數量。選擇較高的值以應對各種場景。
            ransac_thresh (float): RANSAC 演算法的內點門檻值。
        """
        # 選擇 SIFT 是因為精度優先。SIFT 對於尺度、旋轉和光照變化的魯棒性優於 ORB。
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        
        # FLANN (Fast Library for Approximate Nearest Neighbors) 是與 SIFT 這類浮點數描述符配合使用的標準高效匹配器。
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.ransac_thresh = ransac_thresh
        self.global_homography: Optional[np.ndarray] = None

    def _compute_global_homography(self, 
                                   v1_frames: List[np.ndarray], 
                                   video2_path: str, 
                                   sample_size: int = 50) -> bool:
        """
        計算全局 Homography 矩陣 H。
        為了獲得一個極度穩定的 H，我們從兩支影片中聚合大量特徵點來進行計算，
        這能有效濾除單一幀因抖動或物體遮擋造成的雜訊。

        Args:
            v1_frames (List[np.ndarray]): Video1 的5幀查詢群組。
            video2_path (str): Video2 的路徑。
            sample_size (int): 從 Video2 開頭取樣的幀數，以確保有足夠的靜態背景特徵。

        Returns:
            bool: 是否成功計算 Homography。
        """
        print("方案一，步驟 1: 正在計算全局 Homography...")
        
        # --- 從 Video1 的查詢幀中聚合特徵 ---
        kp1_pool, des1_pool_list = [], []
        for frame in v1_frames:
            kp, des = self.sift.detectAndCompute(frame, None)
            if des is not None and len(kp) > 0:
                kp1_pool.extend(kp)
                des1_pool_list.append(des)

        # --- 從 Video2 的樣本幀中聚合特徵 ---
        cap2 = cv2.VideoCapture(video2_path)
        if not cap2.isOpened():
            print(f"錯誤: 無法開啟影片檔案 {video2_path}")
            return False
        
        kp2_pool, des2_pool_list = [], []
        for _ in range(sample_size):
            ret, frame = cap2.read()
            if not ret: break
            kp, des = self.sift.detectAndCompute(frame, None)
            if des is not None and len(kp) > 0:
                kp2_pool.extend(kp)
                des2_pool_list.append(des)
        cap2.release()

        # --- 檢查是否有足夠的特徵 ---
        if not des1_pool_list or not des2_pool_list:
            print("錯誤：無法從影片中提取足夠的 SIFT 特徵。")
            return False

        des1_pool = np.vstack(des1_pool_list)
        des2_pool = np.vstack(des2_pool_list)

        # --- 匹配聚合後的特徵 ---
        # 使用 knnMatch 尋找每個特徵點的最近的2個鄰居，以便後續使用 Lowe's Ratio Test
        matches = self.flann.knnMatch(des1_pool, des2_pool, k=2)

        # --- 應用 Lowe's Ratio Test 篩選優質匹配 ---
        # 這是使用 SIFT 特徵時的黃金標準，能有效剔除模糊和重複紋理造成的錯誤匹配。
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        print(f"聚合了 {len(des1_pool)} 和 {len(des2_pool)} 個特徵點，找到 {len(good_matches)} 個優質匹配。")

        # --- 計算 Homography ---
        # 至少需要4個點才能計算 Homography，但實際上需要更多點才能讓 RANSAC 穩健運作。
        if len(good_matches) > 10:
            src_pts = np.float32([kp1_pool[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2_pool[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # RANSAC 演算法是精髓，它能從含有大量錯誤匹配的點對中，找出符合單一幾何模型的大多數點（內點）。
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
            
            if H is not None:
                self.global_homography = H
                print("全局 Homography 計算成功！")
                return True
        
        print("錯誤：優質匹配點不足，無法計算穩定的 Homography。")
        return False

    def find_best_match(self, v1_frames: List[np.ndarray], video2_path: str) -> Optional[Dict]:
        """
        在 Video2 中尋找與 v1_frames 最匹配的序列。

        Returns:
            Optional[Dict]: 包含匹配資訊的字典，若失敗則返回 None。
        """
        if self.global_homography is None:
            if not self._compute_global_homography(v1_frames, video2_path):
                return None
        
        cap2 = cv2.VideoCapture(video2_path)
        if not cap2.isOpened():
            print(f"錯誤: 無法開啟影片檔案 {video2_path}")
            return None
            
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count2 < len(v1_frames):
            print("錯誤: Video2 的總幀數小於查詢序列的長度。")
            return None
            
        best_match_info = {'index': -1, 'score': -1.0, 'match_frame': None}
        
        print("\n方案一，步驟 2: 正在 Video2 中滑動窗口進行引導式匹配...")
        for j in tqdm(range(frame_count2 - len(v1_frames) + 1), desc="方案一匹配進度"):
            sequence_scores = []
            is_valid_sequence = True
            
            cap2.set(cv2.CAP_PROP_POS_FRAMES, j)
            
            v2_window_frames = []
            for _ in range(len(v1_frames)):
                ret, frame = cap2.read()
                if not ret:
                    is_valid_sequence = False
                    break
                v2_window_frames.append(frame)

            if not is_valid_sequence:
                continue

            for k in range(len(v1_frames)):
                v1_frame = v1_frames[k]
                v2_frame = v2_window_frames[k]
                
                # 核心步驟：使用全局 H 將 v1_frame 校正到 v2 的視角
                warped_v1 = cv2.warpPerspective(v1_frame, self.global_homography, (v2_frame.shape[1], v2_frame.shape[0]))
                
                # 使用歸一化互相關 (NCC) 進行模板匹配評分。NCC 對線性的亮度變化不敏感，非常適合此場景。
                result = cv2.matchTemplate(v2_frame, warped_v1, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                sequence_scores.append(max_val)

            # 穩健評分策略：採用序列中最低的分數作為總分，確保每一幀都匹配良好。
            min_score = min(sequence_scores)
            if min_score > best_match_info['score']:
                best_match_info['score'] = min_score
                best_match_info['index'] = j
                best_match_info['match_frame'] = v2_window_frames[0] # 儲存匹配到的第一幀以供視覺化

        cap2.release()
        
        if best_match_info['index'] != -1:
            return best_match_info
        return None

# ==============================================================================
# 方案二：視覺指紋 (效率與魯棒性導向)
# ==============================================================================
class VisualFingerprintMatcher:
    """
    使用基於深度學習的視覺指紋來同步影片。
    核心策略：
    1. 使用在 ImageNet 上預訓練的 ResNet50 將每一幀壓縮成一個低維度、高資訊含量的「指紋」向量。
    2. 透過計算向量序列間的餘弦相似度來尋找最佳匹配，這是一個非常快速的數學運算。
    3. 實現了快取機制，避免對同一影片重複提取特徵。
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"方案二將使用設備: {self.device}")

        # 載入預訓練的 ResNet50 模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 去掉最後的分類層，只保留到全局平均池化層，其輸出即為我們的特徵向量。
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval() # 設置為評估模式，關閉 Dropout 等

        # 圖像預處理流程，必須與模型訓練時的參數完全一致。
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_feature(self, frame: np.ndarray) -> np.ndarray:
        """從單一幀提取特徵向量。"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)
        
        with torch.no_grad(): # 關閉梯度計算，節省內存和計算資源
            feature = self.feature_extractor(batch_t)
        
        return feature.squeeze().cpu().numpy()

    def _preprocess_video_to_features(self, video_path: str) -> Optional[np.ndarray]:
        """
        將整個影片轉換為特徵向量序列，並實現快取機制。
        """
        cache_path = video_path + ".features.resnet50.npy"
        if os.path.exists(cache_path):
            print(f"從快取載入特徵: {cache_path}")
            return np.load(cache_path)

        print(f"正在為 {video_path} 提取特徵並存入 {cache_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟影片檔案 {video_path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features = []

        for _ in tqdm(range(frame_count), desc=f"提取 {os.path.basename(video_path)} 特徵"):
            ret, frame = cap.read()
            if not ret: break
            feature = self._extract_feature(frame)
            features.append(feature)
        
        cap.release()
        
        if not features:
            print(f"錯誤：無法從 {video_path} 提取任何幀。")
            return None
            
        features_np = np.array(features)
        np.save(cache_path, features_np)
        return features_np

    def find_best_match(self, v1_frames: List[np.ndarray], video2_path: str) -> Optional[Dict]:
        """
        在 Video2 的特徵序列中，尋找與 v1_frames 特徵序列最匹配的部分。
        """
        print("\n方案二，步驟 1: 正在提取查詢序列與目標影片的特徵...")
        v1_features = np.array([self._extract_feature(frame) for frame in v1_frames])
        v2_features = self._preprocess_video_to_features(video2_path)

        if v2_features is None or len(v2_features) < len(v1_features):
            if v2_features is not None:
                 print("錯誤: Video2 的總幀數小於查詢序列的長度。")
            return None
        
        # L2 標準化：這是計算餘弦相似度的預備步驟，標準化後，點積即等價於餘弦相似度。
        v1_features /= np.linalg.norm(v1_features, axis=1, keepdims=True)
        v2_features /= np.linalg.norm(v2_features, axis=1, keepdims=True)

        best_match_info = {'index': -1, 'score': -1.0, 'match_frame': None}
        
        print("\n方案二，步驟 2: 正在特徵序列中滑動窗口進行匹配...")
        num_v2_windows = len(v2_features) - len(v1_features) + 1
        for j in tqdm(range(num_v2_windows), desc="方案二匹配進度"):
            v2_window = v2_features[j : j + len(v1_features)]
            
            # 計算平均餘弦相似度
            avg_similarity = np.mean(np.sum(v1_features * v2_window, axis=1))

            if avg_similarity > best_match_info['score']:
                best_match_info['score'] = avg_similarity
                best_match_info['index'] = j
        
        # 獲取最佳匹配幀以供視覺化
        if best_match_info['index'] != -1:
            cap2 = cv2.VideoCapture(video2_path)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, best_match_info['index'])
            ret, frame = cap2.read()
            if ret:
                best_match_info['match_frame'] = frame
            cap2.release()
            return best_match_info
            
        return None

# ==============================================================================
# 輔助函式與主程式
# ==============================================================================
def load_query_frames(video_path: str, start_frame: int, num_frames: int = 5) -> Optional[List[np.ndarray]]:
    """從指定影片載入查詢幀序列。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟影片檔案 {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame + num_frames > total_frames:
        print(f"錯誤: 請求的幀範圍 ({start_frame} - {start_frame+num_frames}) 超出影片總長度 ({total_frames})。")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("錯誤: 在讀取查詢幀時提前結束。")
            return None
        frames.append(frame)
    cap.release()
    return frames

def visualize_result(query_frame: np.ndarray, match_frame: np.ndarray, start_frame_v1: int, result_info: Dict, method_name: str):
    """將查詢幀與匹配幀並列顯示，並儲存結果圖。"""
    query_frame_rgb = cv2.cvtColor(query_frame, cv2.COLOR_BGR2RGB)
    match_frame_rgb = cv2.cvtColor(match_frame, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'影片對齊結果 ({method_name})', fontsize=16)

    ax1.imshow(query_frame_rgb)
    ax1.set_title(f'Video 1 - 查詢幀 (起始於第 {start_frame_v1} 幀)')
    ax1.axis('off')

    ax2.imshow(match_frame_rgb)
    ax2.set_title(f"Video 2 - 匹配幀 (起始於第 {result_info['index']} 幀)")
    ax2.axis('off')
    
    plt.figtext(0.5, 0.05, f"匹配分數: {result_info['score']:.4f}", ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    output_filename = f'match_result_{method_name}.png'
    plt.savefig(output_filename)
    print(f"\n視覺化結果已儲存至: {output_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="使用兩種方法進行影片時序對齊。")
    parser.add_argument("video1", type=str, help="查詢影片 (Video 1) 的路徑。")
    parser.add_argument("video2", type=str, help="目標影片 (Video 2) 的路徑。")
    parser.add_argument("--start_frame", type=int, default=100, help="從 Video 1 的哪一幀開始選取查詢序列。")
    parser.add_argument("--method", type=str, choices=['local', 'fingerprint', 'all'], default='all', help="選擇使用的方法: 'local' (方案一), 'fingerprint' (方案二), 或 'all' (兩者都執行)。")
    
    args = parser.parse_args()

    # 載入查詢幀
    print(f"正在從 {args.video1} 的第 {args.start_frame} 幀開始載入 5 幀查詢序列...")
    query_frames = load_query_frames(args.video1, args.start_frame)
    if query_frames is None:
        return # 錯誤訊息已在函式中印出

    # 執行方案一
    if args.method in ['local', 'all']:
        print("\n" + "="*50)
        print("--- 執行方案一：強化局部特徵匹配 ---")
        print("="*50)
        matcher1 = EnhancedLocalFeatureMatcher()
        result1 = matcher1.find_best_match(query_frames, args.video2)
        
        if result1:
            print("\n--- 方案一匹配結果 ---")
            print(f"找到最佳匹配！Video1 (始於 {args.start_frame}) <=> Video2 (始於 {result1['index']})")
            print(f"該序列的最低 NCC 匹配分數為: {result1['score']:.4f}")
            visualize_result(query_frames[0], result1['match_frame'], args.start_frame, result1, 'LocalFeature')
        else:
            print("\n--- 方案一匹配失敗 ---")

    # 執行方案二
    if args.method in ['fingerprint', 'all']:
        print("\n" + "="*50)
        print("--- 執行方案二：視覺指紋 ---")
        print("="*50)
        matcher2 = VisualFingerprintMatcher()
        result2 = matcher2.find_best_match(query_frames, args.video2)
        
        if result2:
            print("\n--- 方案二匹配結果 ---")
            print(f"找到最佳匹配！Video1 (始於 {args.start_frame}) <=> Video2 (始於 {result2['index']})")
            print(f"該序列的平均餘弦相似度為: {result2['score']:.4f}")
            visualize_result(query_frames[0], result2['match_frame'], args.start_frame, result2, 'Fingerprint')
        else:
            print("\n--- 方案二匹配失敗 ---")

if __name__ == '__main__':
    main()
