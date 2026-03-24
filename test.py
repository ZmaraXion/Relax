import cv2
import numpy as np

class TemporalNoiseDetector:
    def __init__(self, edge_threshold=30, z_score_threshold=3.0):
        """
        初始化噪點偵測器
        :param edge_threshold: 邊緣梯度的判定閾值 (0-255)，低於此值的區域視為平坦區
        :param z_score_threshold: 用於判定時域噪點突波的 Z-score 閾值
        """
        self.edge_threshold = edge_threshold
        self.z_score_threshold = z_score_threshold
        self.noise_history = []
        
    def _warp_frame(self, frame_t_minus_1, flow_backward):
        """利用反向光流將 t-1 幀對齊到 t 幀"""
        h, w = flow_backward.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        flow_map[:, :, 0] = X + flow_backward[:, :, 0]
        flow_map[:, :, 1] = Y + flow_backward[:, :, 1]
        
        # 使用 INTER_LINEAR 進行亞像素插值，邊緣使用 REPLICATE 避免黑邊干擾
        warped_frame = cv2.remap(frame_t_minus_1, flow_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped_frame

    def _get_flat_region_mask(self, frame_gray):
        """計算圖像的邊緣梯度，回傳平坦區域的遮罩"""
        # 計算 X 與 Y 方向的 Sobel 梯度
        grad_x = cv2.Sobel(frame_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame_gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # 計算梯度強度
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # 梯度小於閾值的區域設為 True (平坦區)
        flat_mask = grad_magnitude < self.edge_threshold
        return flat_mask

    def estimate_frame_noise(self, frame_t, frame_t_minus_1, flow_backward):
        """
        核心方法：計算單幀的時域噪點標準差 (Sigma)
        """
        # 1. 轉為灰階處理以提升效率並專注於亮度噪點 (Luma Noise)
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_t_minus_1 = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 2. 運動補償
        warped_t_minus_1 = self._warp_frame(gray_t_minus_1, flow_backward)

        # 3. 計算平坦區遮罩 (以當前幀為基準)
        flat_mask = self._get_flat_region_mask(gray_t)

        # 4. 計算殘差
        residual = np.abs(gray_t - warped_t_minus_1)

        # 5. 僅提取平坦區域的殘差像素
        valid_residuals = residual[flat_mask]

        if len(valid_residuals) == 0:
            return 0.0

        # 6. 使用 MAD (Median Absolute Deviation) 進行強健估計，排除遮擋區域的極端值
        median_res = np.median(valid_residuals)
        mad = np.median(np.abs(valid_residuals - median_res))
        
        # 轉換為常態分佈的標準差估計值 (1.4826 縮放因子)
        noise_sigma = 1.4826 * mad
        
        self.noise_history.append(noise_sigma)
        return noise_sigma

    def detect_spikes(self):
        """
        分析歷史噪點數據，找出跳動噪點爆發的確切幀索引
        """
        if len(self.noise_history) < 3:
            return []

        history_array = np.array(self.noise_history)
        mean_noise = np.mean(history_array)
        std_noise = np.std(history_array)
        
        if std_noise == 0:
            return []

        # 計算 Z-score 以找出異常峰值
        z_scores = (history_array - mean_noise) / std_noise
        spike_indices = np.where(z_scores > self.z_score_threshold)[0]
        
        # 注意：此處的 index 是從 0 開始，對應到處理的 frame pair (即影片的第 1 幀到第 N 幀)
        # index 0 代表 frame 1 相對於 frame 0 的結果
        return spike_indices.tolist()

# ==========================================
# 模擬執行區塊 (Mock Execution)
# ==========================================
def dummy_ai_optical_flow(frame_t, frame_t_minus_1):
    """
    這是一個佔位函數。未來請替換為您的 AI 光流推論程式碼 (例如 RAFT, GMFlow)。
    請注意：這裡需要的是 Backward Flow (從 t 指向 t-1)。
    此處回傳全零矩陣僅供測試程式順利運行。
    """
    h, w = frame_t.shape[:2]
    # 模擬光流輸出格式：(H, W, 2)，通道 0 為 dx，通道 1 為 dy
    return np.zeros((h, w, 2), dtype=np.float32)

if __name__ == "__main__":
    # 建立測試用的假影片幀 (隨機生成)
    h, w = 480, 640
    frame0 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    frame1 = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) # 模擬高噪點跳動
    frame2 = frame0.copy() 
    
    frames = [frame0, frame1, frame2]
    
    detector = TemporalNoiseDetector(edge_threshold=30, z_score_threshold=1.5)
    
    print("開始逐幀分析時域噪點...")
    for i in range(1, len(frames)):
        frame_t = frames[i]
        frame_t_minus_1 = frames[i-1]
        
        # 1. 獲取光流 (請替換為您的 AI 模型)
        flow_backward = dummy_ai_optical_flow(frame_t, frame_t_minus_1)
        
        # 2. 估算噪點
        sigma = detector.estimate_frame_noise(frame_t, frame_t_minus_1, flow_backward)
        print(f"Frame {i} 噪點標準差 (Sigma): {sigma:.4f}")
        
    # 3. 輸出跳動幀位置
    spikes = detector.detect_spikes()
    print(f"\n分析完成！偵測到跳動噪點的幀索引 (相對第一幀): {spikes}")
