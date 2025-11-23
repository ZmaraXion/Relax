import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # 指定 Matplotlib 使用 Qt5 後端

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QSplitter, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """自定義的 Matplotlib 畫布元件"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # 調整邊距，讓圖表填滿空間
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        super(MplCanvas, self).__init__(self.fig)

class AdvancedHeatmapAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- 1. 系統設置與數據生成 ---
        self.setWindowTitle("Advanced Thermal Analysis & Histogram Tool")
        self.setGeometry(100, 100, 1280, 720) # 加寬視窗以容納直方圖
        
        # 模擬數據
        self.input_rgb, self.input_heatmap = self.generate_dummy_data(640, 480)
        
        # --- 2. 數據預處理 (Pre-processing) ---
        # 影像處理
        self.gray_bg = cv2.cvtColor(self.input_rgb, cv2.COLOR_BGR2GRAY)
        self.gray_bg = cv2.cvtColor(self.gray_bg, cv2.COLOR_GRAY2BGR)
        
        # 數值正規化 (用於 Slider 邏輯)
        self.min_val = np.min(self.input_heatmap)
        self.max_val = np.max(self.input_heatmap)
        # 避免除以零
        range_span = self.max_val - self.min_val if self.max_val != self.min_val else 1.0
        
        norm_heatmap = (self.input_heatmap - self.min_val) / range_span
        self.heatmap_uint8 = (norm_heatmap * 255).astype(np.uint8)
        self.full_color_map = cv2.applyColorMap(self.heatmap_uint8, cv2.COLORMAP_TURBO)

        # --- 3. GUI 佈局設計 (Dashboard Style) ---
        self.init_ui()
        
        # --- 4. 初始化直方圖 ---
        self.plot_histogram()
        
        # 初始畫面更新
        self.update_display()

    def generate_dummy_data(self, w, h):
        """生成模擬數據：包含漸層背景與多個高斯熱點"""
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        
        # RGB 背景
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (xv * 200).astype(np.uint8)
        img[:, :, 1] = (yv * 200).astype(np.uint8)
        img[:, :, 2] = 100
        
        # 複雜一點的 Heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        # 大而弱的背景雜訊
        heatmap += np.random.normal(0, 0.1, (h, w))
        # 熱點 1 (強)
        heatmap += np.exp(-((xv-0.3)**2 + (yv-0.4)**2) / (2*0.05**2)) * 8.0
        # 熱點 2 (中)
        heatmap += np.exp(-((xv-0.7)**2 + (yv-0.7)**2) / (2*0.1**2)) * 4.0
        # 熱點 3 (弱，模擬邊緣情況)
        heatmap += np.exp(-((xv-0.5)**2 + (yv-0.2)**2) / (2*0.02**2)) * 2.5
        
        return img, heatmap

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局採用水平分割：左邊影像，右邊分析面板
        main_layout = QHBoxLayout(central_widget)

        # --- 左側：影像顯示區 ---
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        self.image_label = QLabel("Image Display")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        # 設定影像區的伸縮權重，讓它佔據主要空間
        image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label)
        
        # --- 右側：控制與分析區 ---
        controls_container = QWidget()
        controls_container.setFixedWidth(450) # 固定寬度
        controls_layout = QVBoxLayout(controls_container)
        
        # 1. 直方圖區塊
        hist_group = QGroupBox("Data Distribution (Histogram)")
        hist_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        hist_layout.addWidget(self.canvas)
        
        # 統計數據標籤
        self.stats_label = QLabel("Coverage: 0.00% | Mean Value: 0.00")
        self.stats_label.setStyleSheet("font-weight: bold; color: #333;")
        self.stats_label.setAlignment(Qt.AlignCenter)
        hist_layout.addWidget(self.stats_label)
        
        hist_group.setLayout(hist_layout)
        controls_layout.addWidget(hist_group)

        # 2. 參數控制區塊
        param_group = QGroupBox("Controls")
        param_layout = QVBoxLayout()
        
        # Threshold Slider
        self.thresh_val_label = QLabel("Threshold: 0.00")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 1000)
        self.thresh_slider.setValue(0)
        self.thresh_slider.valueChanged.connect(self.update_display)
        
        # Opacity Slider
        self.opacity_val_label = QLabel("Opacity: 60%")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(60)
        self.opacity_slider.valueChanged.connect(self.update_display)

        param_layout.addWidget(self.thresh_val_label)
        param_layout.addWidget(self.thresh_slider)
        param_layout.addSpacing(15)
        param_layout.addWidget(self.opacity_val_label)
        param_layout.addWidget(self.opacity_slider)
        param_group.setLayout(param_layout)
        
        controls_layout.addWidget(param_group)
        controls_layout.addStretch() # 讓控制項靠上對齊

        # 組合主畫面
        main_layout.addWidget(image_container, 2) # Stretch factor 2
        main_layout.addWidget(controls_container, 1) # Stretch factor 1

    def plot_histogram(self):
        """繪製靜態直方圖並初始化動態紅線"""
        # 攤平陣列以計算直方圖
        data_flat = self.input_heatmap.flatten()
        
        self.canvas.axes.clear()
        # 繪製直方圖 (bins=100 已經足夠精細)
        # log=True 可以幫助看到數量少但數值高的高溫區 (若需要可開啟)
        self.canvas.axes.hist(data_flat, bins=100, color='#1f77b4', alpha=0.7, edgecolor='none')
        
        self.canvas.axes.set_title("Value Frequency Distribution", fontsize=10)
        self.canvas.axes.set_xlabel("Pixel Value", fontsize=8)
        self.canvas.axes.set_ylabel("Count", fontsize=8)
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)
        
        # 初始化那條代表 Threshold 的紅線
        self.vline = self.canvas.axes.axvline(x=self.min_val, color='red', linewidth=2, linestyle='-')
        
        self.canvas.draw()

    @pyqtSlot()
    def update_display(self):
        slider_val = self.thresh_slider.value()
        opacity_val = self.opacity_slider.value() / 100.0
        
        # 計算實際數值
        current_thresh_val = self.min_val + (slider_val / 1000.0) * (self.max_val - self.min_val)
        
        # --- 更新 UI 文字 ---
        self.thresh_val_label.setText(f"Threshold: {current_thresh_val:.4f}")
        self.opacity_val_label.setText(f"Opacity: {int(opacity_val*100)}%")
        
        # --- 圖像處理 (Vectorized) ---
        thresh_uint8 = int((slider_val / 1000.0) * 255)
        mask = self.heatmap_uint8 > thresh_uint8
        
        # 計算統計數據
        active_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        coverage_ratio = (active_pixels / total_pixels) * 100
        
        # 更新統計標籤
        self.stats_label.setText(f"Coverage: {coverage_ratio:.2f}% (Pixels: {active_pixels})")
        
        # 疊加影像
        final_img = self.gray_bg.copy()
        if opacity_val > 0 and active_pixels > 0:
            roi_color = self.full_color_map[mask]
            roi_gray = self.gray_bg[mask]
            blended = cv2.addWeighted(roi_color, opacity_val, roi_gray, 1 - opacity_val, 0)
            final_img[mask] = blended

        # 轉為 QPixmap 顯示
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        h, w, ch = final_img.shape
        qt_img = QImage(final_img.data, w, h, ch * w, QImage.Format_RGB888)
        
        # 使用 setPixmap 並讓圖片保持比例縮放 (選擇性)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # --- 更新直方圖上的紅線 ---
        # 這裡不呼叫 plot_histogram()，而是直接更新線的數據，效能極高
        self.vline.set_xdata([current_thresh_val])
        self.canvas.draw_idle() # 請求重繪 (非阻塞)

    def resizeEvent(self, event):
        # 視窗改變大小時，重新觸發 update_display 以適應圖片縮放
        self.update_display()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedHeatmapAnalyzer()
    window.show()
    sys.exit(app.exec_())
