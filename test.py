import cv2
import numpy as np

class DancingNoiseProfiler:
    def __init__(self, patch_size=5, edge_threshold=15, eps=1e-5):
        """
        :param patch_size: 計算局部變異數的區塊大小 (奇數)
        :param edge_threshold: 邊緣遮罩閾值，用於過濾結構區域
        :param eps: 避免除以零的微小常數
        """
        self.patch_size = patch_size
        self.edge_threshold = edge_threshold
        self.eps = eps

    def _warp_frame(self, frame, flow_backward):
        """利用反向光流對齊影像"""
        h, w = flow_backward.shape[:2]
        flow_map = np.zeros((h, w, 2), dtype=np.float32)
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        flow_map[:, :, 0] = X + flow_backward[:, :, 0]
        flow_map[:, :, 1] = Y + flow_backward[:, :, 1]
        return cv2.remap(frame, flow_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _extract_high_freq_noise(self, gray_frame):
        """提取高頻空間噪點層"""
        smoothed = cv2.GaussianBlur(gray_frame, (5, 5), 1.5)
        return gray_frame - smoothed

    def _compute_local_variance(self, image):
        """利用 BoxFilter 高速計算局部區塊變異數：Var(X) = E[X^2] - (E[X])^2"""
        mean_img = cv2.boxFilter(image, cv2.CV_32F, (self.patch_size, self.patch_size))
        mean_sq_img = cv2.boxFilter(image**2, cv2.CV_32F, (self.patch_size, self.patch_size))
        variance = mean_sq_img - mean_img**2
        return np.maximum(variance, 0) # 確保變異數非負

    def analyze_dancing_noise(self, frame_t, frame_t_minus_1, flow_backward):
        """
        整合性分析：同時輸出熱力圖 (Heatmap) 與全域跳動分數 (Global Score)
        """
        # 1. 轉灰階與噪點層提取
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_prev = cv2.cvtColor(frame_t_minus_1, cv2.COLOR_BGR2GRAY).astype(np.float32)

        noise_t = self._extract_high_freq_noise(gray_t)
        noise_prev = self._extract_high_freq_noise(gray_prev)

        # 2. 運動補償
        warped_noise_prev = self._warp_frame(noise_prev, flow_backward)

        # 3. 生成平坦區域遮罩 (避開邊緣與複雜紋理)
        grad_x = cv2.Sobel(gray_t, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_t, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        flat_mask = (grad_mag < self.edge_threshold).astype(np.float32)

        # 4. 核心計算：STVR (Spatio-Temporal Variance Ratio)
        # 空間變異數 Var(N_t)
        var_spatial = self._compute_local_variance(noise_t)
        
        # 時域殘差變異數 Var(N_t - \hat{N}_{t-1})
        temporal_residual = noise_t - warped_noise_prev
        var_temporal = self._compute_local_variance(temporal_residual)

        # 5. 計算跳動比率 (Dancing Ratio Heatmap)
        # 理論上純跳動噪點 var_temporal 應等於 2 * var_spatial
        expected_temporal_var = 2.0 * var_spatial
        
        # 計算比值，並將範圍截斷在 [0, 1] 之間
        dancing_ratio = var_temporal / (expected_temporal_var + self.eps)
        dancing_ratio = np.clip(dancing_ratio, 0.0, 1.0)

        # 6. 套用遮罩，排除邊緣區的光流誤差干擾
        valid_dancing_heatmap = dancing_ratio * flat_mask

        # 7. 計算全域跳動分數 (Global Dancing Score)
        # 僅統計平坦區域的平均跳動程度
        valid_pixels = np.sum(flat_mask)
        if valid_pixels > 100:
            global_dancing_score = np.sum(valid_dancing_heatmap) / valid_pixels
        else:
            global_dancing_score = 0.0

        # 將熱力圖轉換為視覺化格式 (0-255，Jet Colormap)
        heatmap_vis = np.uint8(valid_dancing_heatmap * 255)
        heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        
        # 將遮罩外區域設為黑色，突顯平坦區的噪點分析
        heatmap_color[flat_mask == 0] = [0, 0, 0]

        return global_dancing_score, heatmap_color

def create_debug_overlay(self, original_frame, heatmap_color, flat_mask, alpha=0.6):
        """
        將跳動噪點熱力圖半透明疊加至原始影像上
        :param original_frame: 當前幀原始彩色影像 (BGR)
        :param heatmap_color: 演算法輸出的彩色熱力圖 (BGR)
        :param flat_mask: 平坦區域遮罩 (0 或 1)
        :param alpha: 熱力圖的不透明度 (0.0 ~ 1.0)
        :return: 疊加完成的除錯影像
        """
        # 1. 確保影像尺寸與格式一致
        if original_frame.shape != heatmap_color.shape:
            heatmap_color = cv2.resize(heatmap_color, (original_frame.shape[1], original_frame.shape[0]))

        # 2. 建立全畫面的 Alpha Blending 基礎圖
        blended_full = cv2.addWeighted(heatmap_color, alpha, original_frame, 1 - alpha, 0)

        # 3. 利用平坦遮罩 (flat_mask) 進行精確合成
        # 確保 mask 為 3 通道以便與彩色影像運算
        mask_3c = np.repeat(flat_mask[:, :, np.newaxis], 3, axis=2).astype(np.float32)
        
        # 僅在遮罩為 1 (平坦區) 的地方顯示疊加結果，遮罩為 0 (邊緣/複雜區) 的地方保留原圖
        debug_overlay = blended_full * mask_3c + original_frame * (1 - mask_3c)
        
        return np.uint8(debug_overlay)
    
# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    # 建立具有隨機動態噪點的模擬畫面
    h, w = 720, 1280
    base_img = np.ones((h, w, 3), dtype=np.uint8) * 128  # 純灰平坦背景
    
    # 模擬強度相同，但位置隨機的跳動噪點
    noise1 = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
    noise2 = np.random.normal(0, 15, (h, w, 3)).astype(np.float32)
    
    frame_prev = np.clip(base_img + noise1, 0, 255).astype(np.uint8)
    frame_t = np.clip(base_img + noise2, 0, 255).astype(np.uint8)
    
    # 假設畫面無移動，光流為零
    dummy_flow = np.zeros((h, w, 2), dtype=np.float32)
    
    profiler = DancingNoiseProfiler(patch_size=7, edge_threshold=20)
    score, heatmap = profiler.analyze_dancing_noise(frame_t, frame_prev, dummy_flow)
    
    print(f"✅ 影格跳動分數 (Global Dancing Score): {score:.4f} (越接近 1 代表跳動越劇烈)")
    # cv2.imwrite("dancing_heatmap.jpg", heatmap) # 儲存熱力圖供視覺檢查
