def get_scielab_evaluator(screen_ppi=109, zoom_percentage=100, view_distance_inch=24):
    """
    工廠函數：根據放大倍率產生對應的 S-CIELAB 計算器
    """
    # 計算縮放因子 (500% -> 5.0)
    zoom_factor = zoom_percentage / 100.0
    
    # 計算等效 PPI
    # 當 zoom 越大，effective_ppi 越小，濾波範圍越小，檢測越嚴格
    effective_ppi = screen_ppi / zoom_factor
    
    # 實例化
    print(f"目前模式: 放大倍率 {zoom_percentage}%, 等效 PPI: {effective_ppi:.1f}")
    return SCIELAB(ppi=effective_ppi, distance_inch=view_distance_inch)

# --- 使用情境 ---
# 情境 A: 使用者正常瀏覽 (100% Zoom) -> 模擬人眼忽略高頻雜訊
scielab_normal = get_scielab_evaluator(zoom_percentage=100)

# 情境 B: 使用者放大檢查 (500% Zoom) -> 模擬人眼看清細節與像素顆粒
scielab_zoomed = get_scielab_evaluator(zoom_percentage=500)
