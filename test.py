import cv2
import numpy as np
import time

class MutualInformationMatcher:
    def __init__(self, bins=256, target_width=640):
        """
        初始化 MI 計算器
        :param bins: 直方圖的階數 (一般 8-bit 灰階圖使用 256)
        :param target_width: 為了加速，將 4K 影像降採樣至此寬度
        """
        self.bins = bins
        self.target_width = target_width

    def _preprocess(self, img):
        """影像預處理：降採樣與灰階轉換"""
        # 降採樣處理 (對於 4K 影像，這步是速度的關鍵)
        if img.shape[1] > self.target_width:
            scale_ratio = self.target_width / img.shape[1]
            target_dim = (self.target_width, int(img.shape[0] * scale_ratio))
            # 使用 INTER_AREA 確保降採樣時保留區域能量特性，抗鋸齒效果最佳
            img = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)
        
        # 若為彩色圖片，轉為灰階以計算單通道聯合分佈
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return img

    def compute_mi(self, img1, img2):
        """
        計算兩張影像的相互資訊 (Mutual Information)
        """
        # 1. 預處理：降維與灰階化
        img1_p = self._preprocess(img1)
        img2_p = self._preprocess(img2)

        # 確保維度一致
        if img1_p.shape != img2_p.shape:
            raise ValueError("Images must have the same dimensions after preprocessing.")

        # 2. 向量化計算 2D 聯合直方圖 (Joint Histogram)
        # 將 2D 影像展平為 1D array，並計算聯合分佈
        hist_2d, _, _ = np.histogram2d(
            img1_p.ravel(), 
            img2_p.ravel(), 
            bins=self.bins, 
            range=[[0, 256], [0, 256]]
        )

        # 3. 計算機率分佈
        # 將頻數轉換為聯合機率 p(x, y)
        pxy = hist_2d / np.sum(hist_2d)
        
        # 計算邊緣機率 p(x) 與 p(y)
        px = np.sum(pxy, axis=1) # 沿 y 軸加總
        py = np.sum(pxy, axis=0) # 沿 x 軸加總

        # 計算期望的獨立機率分佈 p(x) * p(y)
        px_py = px[:, None] * py[None, :]

        # 4. 計算相互資訊 (Mutual Information)
        # 僅計算 p(x,y) > 0 的部分，避免 log(0) 錯誤，同時這也是向量化加速的技巧
        nzs = pxy > 0
        
        # MI 公式: sum( p(x,y) * log2( p(x,y) / (p(x)*p(y)) ) )
        mi = np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))

        # 將結果正規化 (Normalized Mutual Information, NMI) 也是一個選項
        # entropy_x = -np.sum(px[px > 0] * np.log2(px[px > 0]))
        # entropy_y = -np.sum(py[py > 0] * np.log2(py[py > 0]))
        # nmi = 2.0 * mi / (entropy_x + entropy_y)
        
        return mi

# ================= 效能測試 =================
if __name__ == "__main__":
    # 模擬兩張 4K 影像 (3840x2160, 3 channels)
    print("Generating mock 4K images...")
    img_A_4k = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)
    
    # 模擬極端曝光變化 (例如：反轉顏色並加上隨機亮度偏移)
    img_B_4k = 255 - img_A_4k + np.random.randint(-20, 20, (2160, 3840, 3)).astype(np.uint8)
    img_B_4k = np.clip(img_B_4k, 0, 255)

    matcher = MutualInformationMatcher(target_width=640) # 將 4K 降至 640 寬度

    print("Computing MI...")
    start_time = time.time()
    mi_score = matcher.compute_mi(img_A_4k, img_B_4k)
    end_time = time.time()

    print(f"Mutual Information Score: {mi_score:.4f}")
    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")


