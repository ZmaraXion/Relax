import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import deltaE_ciede2000

def analyze_color_degradation(ref_path, dist_path, jnd_threshold=1.0):
    # 1. 讀取影像並正規化到 [0, 1] (假設輸入為 sRGB)
    # OpenCV 讀入是 BGR，需轉 RGB
    img_ref = cv2.imread(ref_path)
    img_dist = cv2.imread(dist_path)
    
    if img_ref is None or img_dist is None:
        raise ValueError("無法讀取影像，請檢查路徑")

    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 2. 轉換至 Lab 色彩空間
    # skimage 的 rgb2lab 預設處理了 Gamma 校正 (sRGB -> Linear -> Lab)
    lab_ref = color.rgb2lab(img_ref)
    lab_dist = color.rgb2lab(img_dist)

    # 3. 計算 CIEDE2000 色差 (Total Error)
    # 這是目前最精確的像素級色差公式
    delta_E = deltaE_ciede2000(lab_ref, lab_dist)
    
    # JND 正規化：數值 1.0 代表剛好可察覺
    jnd_map = delta_E / jnd_threshold

    # 4. 特徵分離：識別「褪色 (Fading)」區域
    # 計算飽和度 (Chroma) C = sqrt(a^2 + b^2)
    C_ref = np.sqrt(lab_ref[:, :, 1]**2 + lab_ref[:, :, 2]**2)
    C_dist = np.sqrt(lab_dist[:, :, 1]**2 + lab_dist[:, :, 2]**2)
    
    # 褪色定義：原圖飽和度 > 處理後飽和度
    # 我們只關心變淡的部分，變濃的部分設為 0
    fading_map = np.maximum(0, C_ref - C_dist)
    
    # 為了讓 fading_map 也能對應 JND 感知，我們粗略將其歸一化
    # 註：單純 Chroma 差值的 JND 約為 1.0 ~ 1.5 Lab 單位
    fading_jnd_map = fading_map / jnd_threshold

    return img_ref, jnd_map, fading_jnd_map

def visualize_results(img_ref, jnd_map, fading_map):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原圖
    axes[0].imshow(img_ref)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 總體色差熱力圖 (Total Delta E)
    im1 = axes[1].imshow(jnd_map, cmap='inferno', vmin=0, vmax=5) # vmax=5 讓嚴重錯誤更明顯
    axes[1].set_title("Perceptual Error (JND Map)\nValues > 1 are visible")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='JND Units')

    # 褪色診斷圖 (Fading Specific)
    im2 = axes[2].imshow(fading_map, cmap='Reds', vmin=0, vmax=5)
    axes[2].set_title("Fading Severity (Saturation Loss)\nRed areas are washed out")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='Chroma Loss (Lab Units)')

    plt.tight_layout()
    plt.show()

# --- 使用範例 ---
# 請替換為你的圖片路徑
# visualize_results(*analyze_color_degradation('original.jpg', 'filtered.jpg'))
