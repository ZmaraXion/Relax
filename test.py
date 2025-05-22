import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 讀取圖片並轉換到 Lab 色彩空間
img = cv2.imread('statue_image.jpg')
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

# 預去噪（雙邊濾波）
img_filtered = img_lab.copy()
img_filtered[:, :, 1:3] = cv2.bilateralFilter(img_filtered[:, :, 1:3], d=9, sigmaColor=20, sigmaSpace=20)

# SLIC 超像素分割
n_segments = 150
segments = slic(img_filtered, n_segments=n_segments, compactness=15, sigma=1, start_label=1)

# 初始化輸出圖片
output_lab = img_lab.copy()

# 自適應 K-Means 聚類
for label in np.unique(segments):
    mask = (segments == label).astype(np.uint8)
    pixels = img_lab[mask > 0][:, 1:3]
    if len(pixels) < 30:  # 忽略小面積超像素
        continue

    # 自適應 K 值選擇（輪廓係數）
    max_k = min(4, len(pixels))
    best_k, best_score = 1, -1
    for k in range(1, max_k + 1):
        if k == 1:
            score = 0.8  # 假設單色區域
        else:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
            score = silhouette_score(pixels, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    # 應用 K-Means 聚類
    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx = labels[np.argmax(counts)]
    a_main, b_main = kmeans.cluster_centers_[dominant_idx]

    # 特殊處理：雕像和背景應為灰色
    if abs(a_main) > 10 or abs(b_main) > 10:  # 若偏離灰色，改用中值
        a_main, b_main = np.median(pixels, axis=0)

    # 應用主導顏色（保留 L 通道）
    output_lab[mask > 0, 1] = a_main
    output_lab[mask > 0, 2] = b_main

# 邊界平滑（Poisson Blending）
mask_all = np.zeros_like(segments, dtype=np.uint8)
for label in np.unique(segments):
    mask = (segments == label).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_all = cv2.bitwise_or(mask_all, mask)
output_lab = cv2.seamlessClone(output_lab, img_lab, mask_all, (img.shape[1]//2, img.shape[0]//2), cv2.NORMAL_CLONE)

# 轉回 RGB 並保存
output_rgb = cv2.cvtColor(output_lab, cv2.COLOR_Lab2RGB)
cv2.imwrite('denoised_statue_image.jpg', output_rgb)

# 質量評估
ssim_score = ssim(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(output_rgb, cv2.COLOR_BGR2GRAY))
print(f"SSIM Score: {ssim_score:.4f}")
