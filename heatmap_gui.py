import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QGroupBox, QSizePolicy,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- 1. 自定義互動式圖片檢視器 (核心升級元件) ---
class InteractiveImageViewer(QGraphicsView):
    """
    支援滾輪縮放、拖曳平移、以及滑鼠位置信號發送的檢視器
    """
    # 定義一個信號，當滑鼠移動時發送 (x, y) 座標
    mouse_moved_signal = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 圖片物件容器
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # 設定互動行為
        self.setDragMode(QGraphicsView.ScrollHandDrag) # 允許左鍵拖曳平移
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # 縮放時以滑鼠為中心
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        
        # 啟用滑鼠追蹤 (即使沒按鍵也能捕捉移動)
        self.setMouseTracking(True)
        
        # 縮放限制
        self.min_zoom = 0.1
        self.max_zoom = 20.0

    def update_image(self, q_img):
        """更新顯示的圖片"""
        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        """處理滾輪縮放"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # 檢查滾輪方向
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # 取得當前縮放比例
        current_scale = self.transform().m11() # m11 是 X 軸縮放係數

        # 檢查是否超過縮放限制
        if (zoom_factor > 1 and current_scale < self.max_zoom) or \
           (zoom_factor < 1 and current_scale > self.min_zoom):
            self.scale(zoom_factor, zoom_factor)

    def mouseMoveEvent(self, event):
        """處理滑鼠移動，計算圖片座標"""
        # 呼叫父類別以保留預設行為 (如拖曳)
        super().mouseMoveEvent(event)
        
        # 將視窗座標轉換為場景(圖片)座標
        scene_pos = self.mapToScene(event.pos())
        
        # 發送信號 (取整數像素座標)
        self.mouse_moved_signal.emit(int(scene_pos.x()), int(scene_pos.y()))


# --- 2. Matplotlib 畫布元件 (維持不變) ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
        super(MplCanvas, self).__init__(self.fig)


# --- 3. 主視窗邏輯 ---
class AdvancedHeatmapAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pro Computer Vision Analyzer (Zoom/Pan/Inspect)")
        self.setGeometry(100, 100, 1400, 800)
        
        # --- 數據準備 ---
        self.w, self.h = 800, 600
        self.input_rgb, self.input_heatmap = self.generate_dummy_data(self.w, self.h)
        
        # 預處理
        self.gray_bg = cv2.cvtColor(self.input_rgb, cv2.COLOR_BGR2GRAY)
        self.gray_bg = cv2.cvtColor(self.gray_bg, cv2.COLOR_GRAY2BGR)
        
        self.min_val = np.min(self.input_heatmap)
        self.max_val = np.max(self.input_heatmap)
        range_span = self.max_val - self.min_val if self.max_val != self.min_val else 1.0
        
        norm_heatmap = (self.input_heatmap - self.min_val) / range_span
        self.heatmap_uint8 = (norm_heatmap * 255).astype(np.uint8)
        self.full_color_map = cv2.applyColorMap(self.heatmap_uint8, cv2.COLORMAP_TURBO)

        # --- GUI 初始化 ---
        self.init_ui()
        self.plot_histogram()
        self.update_display()

    def generate_dummy_data(self, w, h):
        """生成更高解析度的模擬數據"""
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        
        # 模擬電路板或工業表面紋理
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (xv * 50 + np.sin(yv*50)*20).astype(np.uint8) + 50
        img[:, :, 1] = (yv * 50 + np.cos(xv*50)*20).astype(np.uint8) + 50
        img[:, :, 2] = 80
        # 加一點格線模擬
        img[::50, :, :] = 200
        img[:, ::50, :] = 200
        
        # 模擬熱斑
        heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap += np.random.normal(25.0, 1.0, (h, w)) # 基底溫度 25度
        # 異常高溫點
        heatmap += np.exp(-((xv-0.4)**2 + (yv-0.4)**2) / (2*0.02**2)) * 30.0 # +30度
        heatmap += np.exp(-((xv-0.7)**2 + (yv-0.6)**2) / (2*0.05**2)) * 15.0 # +15度
        
        return img, heatmap

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左側區塊：影像顯示 + 資訊列 ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        
        # 1. 資訊列 (Inspector Info)
        info_panel = QGroupBox("Pixel Inspector")
        info_layout = QHBoxLayout()
        self.cursor_info_label = QLabel("Hover over image to inspect values...")
        self.cursor_info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #007ACC;")
        info_layout.addWidget(self.cursor_info_label)
        info_panel.setLayout(info_layout)
        left_layout.addWidget(info_panel)

        # 2. 互動式影像檢視器 (取代原本的 QLabel)
        self.viewer = InteractiveImageViewer()
        self.viewer.mouse_moved_signal.connect(self.on_pixel_hover) # 連接信號
        # 設定邊框美化
        self.viewer.setStyleSheet("border: 1px solid #555; background-color: #1e1e1e;")
        left_layout.addWidget(self.viewer)
        
        # 設定左側伸縮權重
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(left_container, 3) # 佔 3 份寬度

        # --- 右側區塊：控制面板 ---
        controls_container = QWidget()
        controls_container.setFixedWidth(400)
        controls_layout = QVBoxLayout(controls_container)
        
        # 直方圖
        hist_group = QGroupBox("Data Histogram")
        hist_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        hist_layout.addWidget(self.canvas)
        self.stats_label = QLabel("Stats: N/A")
        hist_layout.addWidget(self.stats_label)
        hist_group.setLayout(hist_layout)
        controls_layout.addWidget(hist_group)

        # 滑桿控制
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        
        self.thresh_val_label = QLabel("Threshold: 0.00")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 1000)
        self.thresh_slider.setValue(200) # 預設一點數值
        self.thresh_slider.valueChanged.connect(self.update_display)
        
        self.opacity_val_label = QLabel("Opacity: 60%")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(60)
        self.opacity_slider.valueChanged.connect(self.update_display)

        param_layout.addWidget(self.thresh_val_label)
        param_layout.addWidget(self.thresh_slider)
        param_layout.addSpacing(10)
        param_layout.addWidget(self.opacity_val_label)
        param_layout.addWidget(self.opacity_slider)
        param_group.setLayout(param_layout)
        
        controls_layout.addWidget(param_group)
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_container, 1) # 佔 1 份寬度

    def plot_histogram(self):
        data_flat = self.input_heatmap.flatten()
        self.canvas.axes.clear()
        self.canvas.axes.hist(data_flat, bins=80, color='#1f77b4', alpha=0.7)
        self.canvas.axes.set_title("Temperature Distribution")
        self.canvas.axes.grid(True, linestyle='--', alpha=0.3)
        self.vline = self.canvas.axes.axvline(x=self.min_val, color='r', linewidth=2)
        self.canvas.draw()

    @pyqtSlot(int, int)
    def on_pixel_hover(self, x, y):
        """處理滑鼠懸停事件：讀取數值並更新 UI"""
        # 1. 邊界檢查 (非常重要，否則滑鼠移出圖片邊緣會 Crash)
        if 0 <= x < self.w and 0 <= y < self.h:
            # 2. 獲取原始浮點數值
            val = self.input_heatmap[y, x]
            
            # 3. 判斷是否超過當前門檻 (為了改變文字顏色)
            current_thresh = self.get_current_threshold()
            status_color = "red" if val > current_thresh else "green"
            status_text = "PASS" if val <= current_thresh else "ALERT"
            
            self.cursor_info_label.setText(
                f"Position: ({x}, {y})  |  Value: {val:.4f}  |  "
                f"<span style='color:{status_color}'>[{status_text}]</span>"
            )
        else:
            self.cursor_info_label.setText("Position: (Out of bounds)")

    def get_current_threshold(self):
        slider_val = self.thresh_slider.value()
        return self.min_val + (slider_val / 1000.0) * (self.max_val - self.min_val)

    @pyqtSlot()
    def update_display(self):
        slider_val = self.thresh_slider.value()
        opacity_val = self.opacity_slider.value() / 100.0
        
        current_thresh_val = self.get_current_threshold()
        
        self.thresh_val_label.setText(f"Threshold: {current_thresh_val:.4f}")
        self.opacity_val_label.setText(f"Opacity: {int(opacity_val*100)}%")
        
        # --- 影像處理 ---
        thresh_uint8 = int((slider_val / 1000.0) * 255)
        mask = self.heatmap_uint8 > thresh_uint8
        
        # 統計
        count = np.count_nonzero(mask)
        ratio = (count / mask.size) * 100
        self.stats_label.setText(f"Over Threshold: {ratio:.2f}% ({count} px)")
        
        # 合成
        final_img = self.gray_bg.copy()
        if opacity_val > 0 and count > 0:
            roi_color = self.full_color_map[mask]
            roi_gray = self.gray_bg[mask]
            blended = cv2.addWeighted(roi_color, opacity_val, roi_gray, 1 - opacity_val, 0)
            final_img[mask] = blended

        # 轉成 QImage
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        h, w, ch = final_img.shape
        qt_img = QImage(final_img.data, w, h, ch * w, QImage.Format_RGB888)
        
        # --- 關鍵點：更新 Viewer 中的圖片 ---
        self.viewer.update_image(qt_img)

        # 更新直方圖紅線
        self.vline.set_xdata([current_thresh_val])
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedHeatmapAnalyzer()
    window.show()
    sys.exit(app.exec_())
