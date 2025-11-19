import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage import color
import matplotlib.pyplot as plt

class SCIELAB_Evaluator:
    def __init__(self, ppi=96, viewing_distance_inches=18):
        """
        初始化 S-CIELAB 評估器
        :param ppi: 螢幕像素密度 (Pixels Per Inch), 一般螢幕約 96-110, Retina 約 220+
        :param viewing_distance_inches: 觀看距離 (英吋), 一般桌機約 18-24
        """
        self.ppi = ppi
        self.dist = viewing_distance_inches
        # 計算每個像素代表的視角 (Degrees of visual angle per pixel)
        self.deg_per_pixel = (180 / np.pi) * np.arctan(1.0 / (self.dist * self.ppi))
        
        # 定義 S-CIELAB 的空間濾波器參數 (基於 Zhang & Wandell 1996 論文數據)
        # 格式: [Weight, Spread (sigma) in degrees]
        # Luminance (明度通道) - 較窄，解析度高
        self.params_lum = [
            [0.921, 0.0283], [0.105, 0.133], [-0.108, 4.336]
        ]
        # Red-Green (紅綠通道) - 較寬，解析度低
        self.params_rg = [
            [0.531, 0.0392], [0.330, 0.494]
        ]
        # Blue-Yellow (藍黃通道) - 最寬，解析度最低
        self.params_by = [
            [0.488, 0.0536], [0.371, 0.386]
        ]

    def _generate_kernel(self, params):
        """根據參數生成 2D 高斯核"""
        # 找出最大的 sigma 以決定核的大小
        max_sigma_deg = max([p[1] for p in params])
        max_sigma_px = max_sigma_deg / self.deg_per_pixel
        
        # 核大小通常取 6 * sigma，確保覆蓋足夠範圍
        width = int(np.ceil(6 * max_sigma_px))
        if width % 2 == 0: width += 1 # 保持奇數
        
        x = np.arange(width) - (width // 2)
        X, Y = np.meshgrid(x, x)
        R_sq = X**2 + Y**2
        
        kernel = np.zeros((width, width))
        
        # 疊加多個高斯函數
        for weight, sigma_deg in params:
            sigma_px = sigma_deg / self.deg_per_pixel
            # 避免 sigma 太小導致除以零
            sigma_px = max(sigma_px, 0.5) 
            
            gauss = np.exp(-R_sq / (2 * sigma_px**2))
            gauss = gauss / np.sum(gauss) # 正規化單個高斯
            kernel += weight * gauss
            
        # 確保總和為 1，保持能量守恆
        kernel = kernel / np.sum(kernel)
        return kernel

    def rgb2opp(self, rgb):
        """
        RGB -> XYZ -> Opponent Color Space (Y, rg, by)
        使用標準 sRGB 到 XYZ 矩陣，再轉到對立色空間
        """
        # 1. Linearize sRGB (去除 Gamma)
        mask = rgb > 0.04045
        rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055)**2.4, rgb / 12.92)
        
        # 2. RGB Linear -> XYZ
        # 矩陣資料來源：sRGB standard
        M_rgb2xyz = np.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ])
        xyz = np.dot(rgb_lin, M_rgb2xyz.T)
        
        # 3. XYZ -> Opponent (Zhang & Wandell)
        # 通道: Luminance, Red-Green, Blue-Yellow
        M_xyz2opp = np.array([
            [0.279, 0.72, -0.107],
            [-0.449, 0.29, -0.077],
            [0.086, -0.59, 0.501]
        ])
        opp = np.dot(xyz, M_xyz2opp.T)
        return opp

    def opp2lab(self, opp):
        """
        Opponent -> XYZ -> Lab
        """
        # 1. Opponent -> XYZ
        M_xyz2opp = np.array([
            [0.279, 0.72, -0.107],
            [-0.449, 0.29, -0.077],
            [0.086, -0.59, 0.501]
        ])
        M_opp2xyz = np.linalg.inv(M_xyz2opp)
        xyz = np.dot(opp, M_opp2xyz.T)
        
        # 2. XYZ -> Lab
        # 需注意：skimage 的 xyz2lab 預設 illuminant='D65', observer='2'
        # 我們需要把 xyz 正規化到 [0, 1] 範圍以外嗎？
        # skimage 預期 xyz 是相對值。如果這裡 xyz 數值很小，Lab 轉換會正常運作。
        lab = color.xyz2lab(xyz) 
        return lab

    def compute_scielab(self, img_ref, img_dist):
        """
        計算 S-CIELAB 誤差圖
        img_ref, img_dist: RGB 影像, float [0, 1]
        """
        # 1. 轉換至對立色空間 (Pattern-Color Separable)
        opp_ref = self.rgb2opp(img_ref)
        opp_dist = self.rgb2opp(img_dist)
        
        # 2. 生成空間濾波器
        k_lum = self._generate_kernel(self.params_lum)
        k_rg = self._generate_kernel(self.params_rg)
        k_by = self._generate_kernel(self.params_by)
        
        # 3. 對三個通道分別進行卷積 (Spatial Filtering)
        # 這是 S-CIELAB 模擬人眼模糊的關鍵步驟
        filt_ref = np.zeros_like(opp_ref)
        filt_dist = np.zeros_like(opp_dist)
        
        # 通道 0 (Luminance)
        filt_ref[:,:,0] = convolve(opp_ref[:,:,0], k_lum, mode='nearest')
        filt_dist[:,:,0] = convolve(opp_dist[:,:,0], k_lum, mode='nearest')
        
        # 通道 1 (Red-Green)
        filt_ref[:,:,1] = convolve(opp_ref[:,:,1], k_rg, mode='nearest')
        filt_dist[:,:,1] = convolve(opp_dist[:,:,1], k_rg, mode='nearest')
        
        # 通道 2 (Blue-Yellow)
        filt_ref[:,:,2] = convolve(opp_ref[:,:,2], k_by, mode='nearest')
        filt_dist[:,:,2] = convolve(opp_dist[:,:,2], k_by, mode='nearest')
        
        # 4. 轉回 Lab 空間
        lab_ref_filt = self.opp2lab(filt_ref)
        lab_dist_filt = self.opp2lab(filt_dist)
        
        # 5. 計算 CIEDE2000
        error_map = color.deltaE_ciede2000(lab_ref_filt, lab_dist_filt)
        
        return error_map

# --- 使用範例 (S-CIELAB) ---
def run_scielab_analysis(ref_path, dist_path):
    # 讀取影像
    img_ref = cv2.imread(ref_path)
    img_dist = cv2.imread(dist_path)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 初始化評估器 (假設一般電腦螢幕觀看環境)
    evaluator = SCIELAB_Evaluator(ppi=96, viewing_distance_inches=24)
    
    # 計算 S-CIELAB 熱力圖
    scielab_map = evaluator.compute_scielab(img_ref, img_dist)
    
    # 視覺化
    plt.figure(figsize=(10, 5))
    plt.imshow(scielab_map, cmap='plasma', vmin=0, vmax=5)
    plt.colorbar(label='S-CIELAB Units (Perceptual Error)')
    plt.title("S-CIELAB Error Map\n(Spatial Context Aware)")
    plt.axis('off')
    plt.show()

    return scielab_map

# run_scielab_analysis('original.png', 'filtered.png')
