import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

class SCIELAB:
    def __init__(self, visual_angle_degree=1.0, ppi=96.0, distance_inch=24.0):
        """
        初始化 S-CIELAB 計算器
        
        Args:
            visual_angle_degree: 視角 (通常不需要調整)
            ppi: 螢幕像素密度 (Pixels Per Inch)
            distance_inch: 觀看距離 (Inches)
        """
        # 計算每個視角包含的像素數 (Pixels Per Degree, ppd)
        # 公式: ppd = distance * tan(1 degree) * ppi 
        # 近似為: distance * (pi/180) * ppi
        self.ppd = distance_inch * (np.pi / 180.0) * ppi
        
        # 定義 XYZ 到 對手色空間 (Opponent Color Space) 的轉換矩陣
        # 參考 Zhang & Wandell, 1996
        self.m_xyz_2_opp = np.array([
            [0.279, 0.720, -0.107], # O1: Luminance
            [-0.448, 0.290, 0.157], # O2: Red-Green
            [0.086, -0.590, 0.504]  # O3: Blue-Yellow
        ])
        self.m_opp_2_xyz = np.linalg.inv(self.m_xyz_2_opp)
        
        # 預先計算濾波器 Kernels
        self.kernels = self._generate_kernels()

    def _generate_kernels(self):
        """
        根據 PPD 生成三個通道的空間濾波器 (Spatial Filters)
        這是 S-CIELAB 的靈魂，模擬人眼 CSF。
        使用的是 Gaussian Sum 模型。
        """
        # 參數來自 S-CIELAB 原始論文 (Zhang & Wandell)
        # w: 權重, s: 擴散係數 (spread)
        params = [
            # Channel 1 (Luminance) - 3 Gaussians
            {'w': [0.921, 0.105, -0.108], 's': [0.0283, 0.133, 0.433]},
            # Channel 2 (Red-Green) - 2 Gaussians
            {'w': [0.531, 0.330], 's': [0.0392, 0.494]},
            # Channel 3 (Blue-Yellow) - 2 Gaussians
            {'w': [0.488, 0.371], 's': [0.0536, 0.386]}
        ]
        
        kernels = []
        for p in params:
            # 根據 spread 和 ppd 決定 kernel 大小
            # 3 sigma 覆蓋原則
            max_s = max(p['s'])
            support = int(3.5 * max_s * self.ppd) 
            if support % 2 == 0: support += 1 # 確保是奇數
            
            x, y = np.meshgrid(np.arange(-support//2 + 1, support//2 + 1),
                               np.arange(-support//2 + 1, support//2 + 1))
            r2 = x**2 + y**2
            
            # 混合高斯
            kernel = np.zeros_like(r2, dtype=np.float64)
            for w, s in zip(p['w'], p['s']):
                sigma = s * self.ppd
                kernel += w * np.exp(-r2 / (2 * sigma**2))
            
            # 正規化：確保亮度不變
            kernel /= np.sum(kernel)
            kernels.append(kernel)
            
        return kernels

    def _rgb_to_xyz(self, img_rgb):
        """
        sRGB -> Linear RGB -> XYZ
        注意：這裡假設輸入是標準 sRGB [0, 255]
        """
        # 1. 正規化到 [0, 1]
        img = img_rgb.astype(np.float64) / 255.0
        
        # 2. 反 Gamma 校正 (sRGB -> Linear RGB)
        mask = img > 0.04045
        img[mask] = ((img[mask] + 0.055) / 1.055) ** 2.4
        img[~mask] = img[~mask] / 12.92
        
        # 3. Linear RGB -> XYZ (D65)
        # OpenCV 的 COLOR_RGB2XYZ 使用的是標準 sRGB 矩陣
        img_xyz = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2XYZ)
        return img_xyz.astype(np.float64)

    def _xyz_to_cielab(self, img_xyz):
        """
        XYZ -> CIELAB
        為了方便，我們可以使用 OpenCV 的實作，但需要注意 OpenCV 對 XYZ 的範圍定義。
        這裡我們手動實作以確保精度與白點控制 (D65)。
        """
        # Reference White (D65)
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        
        x = img_xyz[:, :, 0] / Xn
        y = img_xyz[:, :, 1] / Yn
        z = img_xyz[:, :, 2] / Zn
        
        def f(t):
            delta = 6/29
            mask = t > delta**3
            res = np.zeros_like(t)
            res[mask] = np.cbrt(t[mask])
            res[~mask] = t[~mask] / (3 * delta**2) + 4/29
            return res
            
        fx, fy, fz = f(x), f(y), f(z)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)

    def process_image(self, img_rgb):
        """
        S-CIELAB 的核心處理流程：RGB -> XYZ -> Opponent -> Filter -> XYZ -> LAB
        """
        # 1. RGB to XYZ
        img_xyz = self._rgb_to_xyz(img_rgb)
        
        # 2. XYZ to Opponent Space
        # 矩陣乘法: (H, W, 3) dot (3, 3).T
        img_opp = np.dot(img_xyz, self.m_xyz_2_opp.T)
        
        # 3. Spatial Filtering (Separable convolution would be faster, using 2D for clarity)
        img_opp_filtered = np.zeros_like(img_opp)
        for i in range(3):
            # 使用 'same' 保持尺寸，'symm' 處理邊界以減少邊緣偽影
            img_opp_filtered[:, :, i] = convolve2d(
                img_opp[:, :, i], self.kernels[i], mode='same', boundary='symm'
            )
            
        # 4. Filtered Opponent to XYZ
        img_xyz_filtered = np.dot(img_opp_filtered, self.m_opp_2_xyz.T)
        
        # 5. XYZ to CIELAB
        img_lab = self._xyz_to_cielab(img_xyz_filtered)
        
        return img_lab

    def compute_difference(self, img1_path, img2_path):
        """
        計算兩張圖的 S-CIELAB Delta E
        """
        # 讀取影像 (OpenCV 讀入為 BGR，需轉為 RGB)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("無法讀取影像")
            
        if img1.shape != img2.shape:
             raise ValueError(f"影像尺寸不符: {img1.shape} vs {img2.shape}")

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 取得經過空間濾波的 LAB 數值
        lab1 = self.process_image(img1)
        lab2 = self.process_image(img2)
        
        # 計算 Delta E (Euclidean distance in LAB space)
        # 這是最原始的 Delta E 1976，若需要更精準可改用 Delta E 2000
        delta_e = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))
        
        return delta_e

# --- 使用範例 ---
if __name__ == "__main__":
    # 設定情境：一般桌上型螢幕，距離 50cm (約 20 inch)，96 DPI
    scielab = SCIELAB(ppi=96, distance_inch=19.7)
    
    # 假設你有兩張圖片 'ref.png' 和 'dist.png'
    # 為了演示，我們生成兩張假圖
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite('ref.png', img1)
    # 加一點高頻雜訊
    noise = np.random.normal(0, 20, img1.shape).astype(np.int16)
    img2 = np.clip(img1 + noise, 0, 255).astype(np.uint8)
    cv2.imwrite('dist.png', img2)

    try:
        diff_map = scielab.compute_difference('ref.png', 'dist.png')
        print(f"平均 S-CIELAB Error: {np.mean(diff_map):.4f}")
        print(f"最大 S-CIELAB Error: {np.max(diff_map):.4f}")
        
        # 視覺化誤差圖
        diff_viz = (diff_map / np.max(diff_map) * 255).astype(np.uint8)
        cv2.imwrite('scielab_diff.png', cv2.applyColorMap(diff_viz, cv2.COLORMAP_JET))
        print("誤差熱力圖已保存為 scielab_diff.png")
        
    except Exception as e:
        print(f"Error: {e}")
        
