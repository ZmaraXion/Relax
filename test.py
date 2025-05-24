import os
import cv2
import matplotlib.pyplot as plt
import pywt
import numpy as np

def estimate_local_noise(channel, window_size=5):
    """估計局部噪點標準差"""
    # 使用高斯模糊近似局部方差
    blurred = cv2.GaussianBlur(channel.astype(np.float32), (window_size, window_size), 0)
    blurred_square = cv2.GaussianBlur(channel.astype(np.float32)**2, (window_size, window_size), 0)
    variance = blurred_square - blurred**2
    variance = np.maximum(variance, 0)  # 避免負值
    return np.sqrt(variance)

def adaptive_nlm(channel, guide, h_base=10, k=1.5, template_size=7, search_size=21):
    """自適應NLM，使用分塊近似"""
    # 估計局部噪點
    sigma = estimate_local_noise(channel)
    h_map = k * sigma + h_base  # 動態h，h_base避免過小
    h_map = np.clip(h_map, 5, 50)  # 限制h範圍

    # 分塊處理以模擬像素級h
    block_size = 32
    height, width = channel.shape
    denoised = np.zeros_like(channel, dtype=np.float32)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            i_end = min(i + block_size, height)
            j_end = min(j + block_size, width)
            block = channel[i:i_end, j:j_end]
            h = np.mean(h_map[i:i_end, j:j_end])  # 使用塊平均h
            block_denoised = cv2.fastNlMeansDenoising(
                block, h=float(h), templateWindowSize=template_size, searchWindowSize=search_size
            )
            denoised[i:i_end, j:j_end] = block_denoised
    
    # 應用引導濾波增強邊緣
    guided = cv2.ximgproc.guidedFilter(
        guide=guide, src=denoised, radius=5, eps=500
    )
    return guided.astype(np.uint8)

def estimate_local_noise2(channel, window_size=5):
    """估計局部噪點標準差"""
    blurred = cv2.GaussianBlur(channel.astype(np.float32), (window_size, window_size), 0)
    blurred_square = cv2.GaussianBlur(channel.astype(np.float32)**2, (window_size, window_size), 0)
    variance = blurred_square - blurred**2
    variance = np.maximum(variance, 0)
    return np.sqrt(variance)

def generate_color_mask(a, b, threshold=40, kernel_size=3):
    """生成色彩保護遮罩（Lab空間）"""
    # Lab空間的a、b範圍為[-128, 127]，偏移至[0, 255]計算飽和度
    a_shifted = a.astype(np.float32) + 128
    b_shifted = b.astype(np.float32) + 128
    saturation = np.sqrt(a_shifted**2 + b_shifted**2)
    # 高通濾波檢測高飽和度邊緣
    sobel_x = cv2.Sobel(saturation, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(saturation, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    mask = (edge_magnitude > threshold / 255.0).astype(np.float32)
    return cv2.GaussianBlur(mask, (5, 5), 0)  # 平滑遮罩邊界

def adaptive_nlm2(channel, guide, mask, h_base=8, k=1.2, template_size=5, search_size=15):
    """自適應NLM，融入色彩保護遮罩"""
    sigma = estimate_local_noise2(channel)
    h_map = k * sigma + h_base
    h_map = h_map * (1 - mask)  # 在高飽和度區域降低h
    h_map = np.clip(h_map, 3, 40)  # 適應Lab空間的較小範圍
    
    block_size = 32
    height, width = channel.shape
    denoised = np.zeros_like(channel, dtype=np.float32)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            i_end = min(i + block_size, height)
            j_end = min(j + block_size, width)
            block = channel[i:i_end, j:j_end]
            h = np.mean(h_map[i:i_end, j:j_end])
            block_denoised = cv2.fastNlMeansDenoising(
                block, h=float(h), templateWindowSize=template_size, searchWindowSize=search_size
            )
            denoised[i:i_end, j:j_end] = block_denoised
    
    guided = cv2.ximgproc.guidedFilter(
        guide=guide, src=denoised, radius=3, eps=50
    )
    return guided.astype(np.float32)  # 保持float32以支持Lab範圍

def color_correction(denoised, original):
    """色彩校正，恢復飽和度"""
    mu_orig, std_orig = np.mean(original), np.std(original)
    mu_denoised, std_denoised = np.mean(denoised), np.std(denoised)
    if std_denoised > 0:  # 避免除零
        corrected = denoised * (std_orig / std_denoised) + (mu_orig - mu_denoised)
    else:
        corrected = denoised
    return np.clip(corrected, 0, 255).astype(np.uint8)

def histogram_matching(denoised, original):
    """直方圖匹配恢復色彩分佈"""
    # 計算原始和去噪後的直方圖與CDF
    hist_orig, bins = np.histogram(original.flatten(), 256, [0, 256])
    hist_denoised, _ = np.histogram(denoised.flatten(), 256, [0, 256])
    cdf_orig = hist_orig.cumsum() / hist_orig.sum()
    cdf_denoised = hist_denoised.cumsum() / hist_denoised.sum()
    
    # 建立映射表
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.argmin(np.abs(cdf_denoised[i] - cdf_orig))
        mapping[i] = j
    
    # 應用映射
    corrected = cv2.LUT(denoised, mapping)
    return corrected

img_path = r".\NOISY_SRGB_010_.png"
img = cv2.imread(img_path)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

l, a, b = cv2.split(lab)
lab_denoised_ada = lab.copy()
lab_denoised_ada[:, :, 1:2] = np.expand_dims(adaptive_nlm(a.copy(), l.copy(), h_base=10, k=1.5), axis=-1)
lab_denoised_ada[:, :, 2:3] = np.expand_dims(adaptive_nlm(b.copy(), l.copy(), h_base=10, k=1.5), axis=-1)

color_mask = generate_color_mask(a.copy(), b.copy(), threshold=30)
lab_denoised_ada_mask = lab.copy()
lab_denoised_ada_mask[:, :, 1:2] = np.expand_dims(adaptive_nlm2(a.copy(), l.copy(), color_mask, h_base=10, k=1.5), axis=-1)
lab_denoised_ada_mask[:, :, 2:3] = np.expand_dims(adaptive_nlm2(b.copy(), l.copy(), color_mask, h_base=10, k=1.5), axis=-1)

# Merge and convert back to RGB
cv2.imwrite(r'.\output_NLM_ada_lab.jpg', cv2.cvtColor(lab_denoised_ada, cv2.COLOR_Lab2BGR))
cv2.imwrite(r'.\output_NLM_ada_lab_mask.jpg', cv2.cvtColor(lab_denoised_ada_mask, cv2.COLOR_Lab2BGR))

# corr = histogram_matching(cv2.cvtColor(lab_denoised_ada, cv2.COLOR_Lab2BGR), img)
# cv2.imwrite(r'.\output_NLM_ada_lab_corr.jpg', corr)
