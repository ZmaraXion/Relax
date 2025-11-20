import cv2
import numpy as np

class ChromaDenoiser:
    def __init__(self, radius=8, eps=0.02, dark_boost=1.5):
        """
        初始化參數
        :param radius: 導向濾波的半徑 (控制平滑範圍)
        :param eps: 導向濾波的正則化參數 (控制對邊緣的敏感度，越小越保持邊緣)
        :param dark_boost: 暗部去噪強度的增益係數 (大於1代表暗部去噪更強)
        """
        self.radius = radius
        # eps 傳入通常是針對 0-1 範圍，OpenCV 內部運算可能需要調整
        self.eps = eps  
        self.dark_boost = dark_boost

    def _gamma_correction(self, img, gamma=2.2, inverse=False):
        """ 處理 Gamma 校正與反校正 """
        if inverse:
            # Linearize: sRGB -> Linear RGB
            return np.power(img, gamma)
        else:
            # Re-apply Gamma: Linear RGB -> sRGB
            return np.power(img, 1.0 / gamma)

    def process(self, rgb_img):
        """
        核心處理流程
        :param rgb_img: 輸入 RGB 影像 (0-255, uint8)
        :return: 處理後的 RGB 影像
        """
        # 1. 前處理：歸一化到 [0, 1] 並轉為浮點數
        img_float = rgb_img.astype(np.float32) / 255.0

        # 2.【關鍵點一】Gamma Linearization (線性化)
        # 在線性空間處理噪點物理上更準確，避免色彩偏移
        img_linear = self._gamma_correction(img_float, inverse=True)

        # 3. 色彩空間轉換 (使用 LAB)
        # L: 亮度 (結構資訊), A/B: 色度 (噪點資訊)
        lab = cv2.cvtColor(img_linear, cv2.COLOR_RGB2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 4.【關鍵點二】建立導引圖 (Guidance Image)
        # 使用 L 通道作為導引，因為它包含最準確的結構資訊
        guidance = l_channel

        # 5.【關鍵點三】建立暗部自適應權重 Mask (Perceptual Masking)
        # 計算亮度權重：越暗的地方 (L 值小)，權重越大
        # L channel 在 OpenCV float 下通常範圍較大，我們先 normalize 觀測一下
        # 假設 L 範圍約 0-100
        l_norm = l_channel / 100.0
        
        # 權重公式：(1 - L) * boost。 L越小，weight越高。
        # 這裡做一個簡單的 Sigmoid 或者是線性反轉
        adaptive_weight = 1.0 + (1.0 - l_norm) * (self.dark_boost - 1.0)
        adaptive_weight = np.clip(adaptive_weight, 1.0, self.dark_boost)
        
        # 為了應用這個 weight，我們調整 eps (epsilon)。
        # Epsilon 越小，邊緣保護越好但去噪越弱；Epsilon 越大，越模糊。
        # 我們希望暗部更模糊一點 (去噪強)，所以暗部 eps 要變大？
        # 其實更直接的做法是：做兩次濾波，或是在混合時調整。
        # 為了實作簡潔且高效，我們這裡採用「動態混合」策略。
        
        # 6. 執行 Guided Filter (核心去噪)
        # 針對 A 和 B 通道分別濾波
        # 使用 opencv-contrib 的 guidedFilter
        
        # 注意：eps 在 OpenCV 中是像素值差的平方。
        # 如果數值範圍是 0-100 (Lab)，eps 需要設大一點，例如 1000
        # 如果我們把 Lab 轉回 0-1 區間處理會比較直觀
        
        a_norm = a_channel # Lab 的 ab 範圍通常在 -128 到 127 之間 (視實作而定)
        b_norm = b_channel 
        
        # 為了通用性，這裡直接用 guidedFilter，OpenCV 會自動處理
        # radius: 濾波半徑
        # eps: 這裡需要根據 Lab 的數值範圍調整。OpenCV Lab float 範圍 L:0-100, a,b: -127~127
        # 經驗值：對於 float Lab, eps 設為 10~50 左右比較合適 (對應 0-1 的 0.001~0.005)
        # 我們使用輸入的 self.eps (假設針對 0-1) 乘上數值範圍的平方比例
        real_eps = (self.eps * 100) ** 2 

        a_clean = cv2.ximgproc.guidedFilter(guide=guidance, src=a_channel, radius=self.radius, eps=real_eps)
        b_clean = cv2.ximgproc.guidedFilter(guide=guidance, src=b_channel, radius=self.radius, eps=real_eps)

        # 7.【關鍵點三實作】基於亮度的自適應混合 (Blending)
        # 在暗部使用 clean 版本，在極亮部可以稍微保留一點原始細節(如果需要)
        # 這裡我們直接將 clean 的結果應用，但如果你發現亮部邊緣有溢色，可以用 mask 混合回來
        # 這裡示範如何用 Mask 混合：
        
        # 製作混合 Mask: 暗部全用 clean，亮部 80% clean (視情況)
        # 為了最大化去彩噪，這裡我們假設全圖都要去，但你可以打開下面的註解做混合
        
        # mask = np.clip(1.0 - l_norm, 0.0, 1.0) # 簡單的暗部 mask
        # a_final = a_clean * mask + a_channel * (1 - mask)
        # b_final = b_clean * mask + b_channel * (1 - mask)
        
        # 目前策略：直接採用濾波結果，因為 Guided Filter 本身就有保邊功能
        a_final = a_clean
        b_final = b_clean

        # 8. 合併通道
        lab_clean = cv2.merge([l_channel, a_final, b_final])

        # 9. 轉回 RGB Linear
        img_clean_linear = cv2.cvtColor(lab_clean, cv2.COLOR_Lab2RGB)

        # 10. 轉回 Gamma (sRGB)
        img_clean_srgb = self._gamma_correction(img_clean_linear, inverse=False)

        # 11. 格式處理 (Clip & Convert to uint8)
        img_clean_srgb = np.clip(img_clean_srgb * 255, 0, 255).astype(np.uint8)

        return img_clean_srgb

# --- 使用範例 ---
if __name__ == "__main__":
    # 讀取圖片 (模擬含彩噪的 ISP 輸出)
    # 請替換成你自己的圖片路徑
    input_img = cv2.imread("noisy_isp_output.jpg") 
    
    if input_img is not None:
        # 參數調整建議：
        # radius: 根據噪點顆粒大小調整。如果是大色斑，設為 16 或 32。細小噪點設為 4-8。
        # eps: 控制邊緣保護。數值越小，越不願意跨越邊緣模糊 (保護越好但去噪越弱)。
        denoiser = ChromaDenoiser(radius=8, eps=0.01, dark_boost=1.2)
        
        result_img = denoiser.process(input_img)

        # 顯示比較
        comparison = np.hstack((input_img, result_img))
        cv2.imshow("Original vs Chroma Denoised", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 儲存結果
        cv2.imwrite("result.png", result_img)
    else:
        print("請提供有效的影像路徑進行測試。")
