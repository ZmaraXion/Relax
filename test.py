import cv2
import numpy as np

def estimate_local_noise(channel, window_size=5):
    """估計局部噪點標準差"""
    blurred = cv2.GaussianBlur(channel.astype(np.float32), (window_size, window_size), 0)
    blurred_square = cv2.GaussianBlur(channel.astype(np.float32)**2, (window_size, window_size), 0)
    variance = blurred_square - blurred**2
    variance = np.maximum(variance, 0)
    return np.sqrt(variance)

def generate_color_mask(a, b, threshold=30, kernel_size=3):
    """生成色彩保護遮罩（Lab空間）"""
    a_shifted = a.astype(np.float32) + 128
    b_shifted = b.astype(np.float32) + 128
    saturation = np.sqrt(a_shifted**2 + b_shifted**2)
    sobel_x = cv2.Sobel(saturation, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(saturation, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    mask = (edge_magnitude > threshold / 255.0).astype(np.float32)
    return cv2.GaussianBlur(mask, (5, 5), 0)

def adaptive_nlm(channel, guide, mask, h_base=8, k=1.2, template_size=5, search_size=15):
    """自適應NLM，融入色彩保護遮罩"""
    sigma = estimate_local_noise(channel)
    h_map = k * sigma + h_base
    h_map = h_map * (1 - mask)
    h_map = np.clip(h_map, 3, 40)
    
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
    
    return denoised

def multiscale_nlm(channel, guide, mask, scales=[(11, 41, 10, 1.5), (5, 15, 8, 1.2)]):
    """多尺度NLM：大窗口處理低頻，小窗口精細化"""
    denoised = channel.astype(np.float32)
    for template_size, search_size, h_base, k in scales:
        denoised = adaptive_nlm(denoised, guide, mask, h_base, k, template_size, search_size)
        # 應用引導濾波每階段增強邊緣
        denoised = cv2.ximgproc.guidedFilter(
            guide=guide, src=denoised, radius=3, eps=50
        )
    return denoised.astype(np.float32)

def local_histogram_matching(denoised, original, block_size=32):
    """局部直方圖匹配"""
    height, width = denoised.shape
    corrected = np.zeros_like(denoised, dtype=np.float32)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            i_end = min(i + block_size, height)
            j_end = min(j + block_size, width)
            block_denoised = denoised[i:i_end, j:j_end] + 128
            block_orig = original[i:i_end, j:j_end] + 128
            hist_orig, bins = np.histogram(block_orig.flatten(), 256, [0, 256])
            hist_denoised, _ = np.histogram(block_denoised.flatten(), 256, [0, 256])
            cdf_orig = hist_orig.cumsum() / hist_orig.sum()
            cdf_denoised = hist_denoised.cumsum() / hist_denoised.sum()
            mapping = np.zeros(256, dtype=np.float32)
            for k in range(256):
                m = np.argmin(np.abs(cdf_denoised[k] - cdf_orig))
                mapping[k] = m
            corrected[i:i_end, j:j_end] = cv2.LUT(block_denoised.astype(np.uint8), mapping) - 128
    
    corrected = cv2.GaussianBlur(corrected, (5, 5), 0)  # 平滑塊邊界
    return corrected

def main(input_path, output_path):
    # 讀取影像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("無法讀取影像")

    # RGB to Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    a_orig, b_orig = a.copy(), b.copy()

    # 生成色彩保護遮罩
    color_mask = generate_color_mask(a, b, threshold=30)

    # 多尺度NLM
    scales = [(11, 41, 10, 1.5), (5, 15, 8, 1.2)]  # 大窗口低頻，小窗口高頻
    a_denoised = multiscale_nlm(a, l, color_mask, scales)
    b_denoised = multiscale_nlm(b, l, color_mask, scales)

    # 局部直方圖匹配
    a_corrected = local_histogram_matching(a_denoised, a_orig, block_size=32)
    b_corrected = local_histogram_matching(b_denoised, b_orig, block_size=32)

    # 值域檢查
    a_corrected = np.clip(a_corrected, -128, 127)
    b_corrected = np.clip(b_corrected, -128, 127)

    # 合併並轉回RGB
    lab_denoised = cv2.merge((l, a_corrected.astype(np.uint8), b_corrected.astype(np.uint8)))
    result = cv2.cvtColor(lab_denoised, cv2.COLOR_Lab2BGR)

    # 保存結果
    cv2.imwrite(output_path, result)
    return result

# 執行
input_path = 'input.jpg'
output_path = 'output_adaptive_nlm_lab_multiscale.jpg'
main(input_path, output_path)



import pywt
def wavelet_lowfreq_denoise(channel, wavelet='db4', level=2):
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size))
    coeffs[0] = pywt.threshold(coeffs[0], threshold * 0.5, mode='soft')  # 低頻子帶輕度閾值
    return pywt.waverec2(coeffs, wavelet)
a_denoised = wavelet_lowfreq_denoise(a)
b_denoised = wavelet_lowfreq_denoise(b)
