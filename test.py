#!/usr/bin/env python3
"""
Noise Patch Sampler
===================
從少量噪聲影像的平坦區域抽取噪聲 patch，建立噪聲資料庫，
再將噪聲 patch 貼回其他乾淨影像的平坦區域。

適用場景：
  - Impulse noise 經 ISP pipeline 後的方向性大顆粒團狀噪聲
  - 伴隨 over/undershoot ringing
  - 僅需處理平坦區域
  - 極少量噪聲樣本、無配對乾淨圖

用法：
  # 1. 從噪聲影像抽取噪聲 patch，建立資料庫
  python noise_patch_sampler.py extract -i noisy_images/ -o noise_db.npz

  # 2. 將噪聲資料庫套用到乾淨影像
  python noise_patch_sampler.py synthesize -i clean_images/ -d noise_db.npz -o output/

  # 3. 視覺化噪聲資料庫內容
  python noise_patch_sampler.py visualize -d noise_db.npz -o viz/

  # 4. 分析噪聲統計特性
  python noise_patch_sampler.py analyze -i noisy_images/

需求：
  pip install numpy opencv-python
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIG = {
    # --- 噪聲抽取 ---
    "patch_size": 96,          # patch 邊長（像素），需涵蓋完整 blob 結構
    "overlap_ratio": 0.5,      # patch 重疊率（0.5 = 50%）
    "median_kernel": 21,       # median filter kernel 大小（奇數），需 >= 2*blob_diameter+1
    "flat_var_threshold": 80,  # 平坦區域判定閾值：局部方差 < 此值視為平坦
    "n_brightness_bins": 4,    # 亮度分 bin 數量
    "min_flat_ratio": 0.7,     # patch 內平坦像素佔比需 >= 此值才納入

    # --- 噪聲合成 ---
    "blend_margin": 16,        # patch 邊緣 feathered blending 寬度（像素）
    "intensity_jitter": 0.1,   # 噪聲振幅隨機抖動比例 (±10%)
    "allow_flip": True,        # 允許隨機翻轉 patch（增加多樣性）
    "allow_rotate": True,      # 允許 90° 旋轉（若噪聲無嚴格方向性可開啟）
}


# ============================================================
# Core: Flat Region Detection
# ============================================================

def compute_local_variance(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    """計算灰階影像的局部方差圖。"""
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.maximum(sq_mean - mean ** 2, 0)
    return variance


def detect_flat_mask(img: np.ndarray, threshold: float, ksize: int = 15) -> np.ndarray:
    """
    偵測平坦區域。
    回傳 boolean mask，True = 平坦。
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    var_map = compute_local_variance(gray, ksize)
    return var_map < threshold


def auto_flat_threshold(img: np.ndarray, percentile: float = 30.0) -> float:
    """
    自動估計平坦區域閾值。
    取局部方差分佈的指定百分位數作為閾值。
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    var_map = compute_local_variance(gray, ksize=15)
    threshold = np.percentile(var_map, percentile)
    return float(threshold)


# ============================================================
# Core: Noise Residual Extraction
# ============================================================

def extract_noise_residual(
    img: np.ndarray,
    median_kernel: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 median filter 估計乾淨信號，取 residual 作為噪聲估計。

    Median filter 對 impulse noise 天然魯棒，是無配對情況下的最佳選擇。
    Kernel 大小需 >= 2 * blob_diameter + 1。

    Returns:
        residual: float32, 噪聲殘差（可為負值）
        pseudo_clean: uint8, median filter 輸出的偽乾淨影像
    """
    if median_kernel % 2 == 0:
        median_kernel += 1

    if len(img.shape) == 3:
        pseudo_clean = np.stack([
            cv2.medianBlur(img[:, :, c], median_kernel)
            for c in range(img.shape[2])
        ], axis=-1)
    else:
        pseudo_clean = cv2.medianBlur(img, median_kernel)

    residual = img.astype(np.float32) - pseudo_clean.astype(np.float32)
    return residual, pseudo_clean


# ============================================================
# Core: Patch Extraction
# ============================================================

def extract_noise_patches(
    img: np.ndarray,
    config: dict
) -> dict[int, list[np.ndarray]]:
    """
    從單張噪聲影像的平坦區域抽取噪聲 patch，按亮度分 bin。

    Returns:
        patches_by_bin: {bin_index: [patch_array, ...]}
            每個 patch 為 float32，shape = (patch_size, patch_size, C)
    """
    ps = config["patch_size"]
    overlap = int(ps * config["overlap_ratio"])
    step = ps - overlap
    n_bins = config["n_brightness_bins"]
    min_flat = config["min_flat_ratio"]

    # 1. 偵測平坦區域
    flat_mask = detect_flat_mask(img, config["flat_var_threshold"])

    # 2. 提取噪聲殘差
    residual, pseudo_clean = extract_noise_residual(img, config["median_kernel"])

    # 3. 計算亮度圖（用偽乾淨影像）
    if len(pseudo_clean.shape) == 3:
        brightness = cv2.cvtColor(pseudo_clean, cv2.COLOR_BGR2GRAY)
    else:
        brightness = pseudo_clean

    # 4. 滑動窗口抽取 patch
    h, w = img.shape[:2]
    patches_by_bin: dict[int, list[np.ndarray]] = {i: [] for i in range(n_bins)}

    bin_edges = np.linspace(0, 256, n_bins + 1)

    for y in range(0, h - ps + 1, step):
        for x in range(0, w - ps + 1, step):
            # 檢查此 patch 區域是否足夠平坦
            patch_flat = flat_mask[y:y + ps, x:x + ps]
            flat_ratio = np.mean(patch_flat)
            if flat_ratio < min_flat:
                continue

            # 計算此 patch 的平均亮度 → 決定 bin
            patch_brightness = np.mean(brightness[y:y + ps, x:x + ps])
            bin_idx = int(np.clip(
                np.searchsorted(bin_edges[1:], patch_brightness),
                0, n_bins - 1
            ))

            # 抽取噪聲 patch
            noise_patch = residual[y:y + ps, x:x + ps].copy()
            patches_by_bin[bin_idx].append(noise_patch)

    return patches_by_bin


def build_noise_database(
    image_paths: list[str],
    config: dict
) -> dict:
    """
    從多張噪聲影像建立噪聲 patch 資料庫。

    Returns:
        db: {
            "config": dict,
            "patches_bin_0": np.ndarray,  # shape (N, ps, ps, C)
            "patches_bin_1": np.ndarray,
            ...
            "stats": dict
        }
    """
    n_bins = config["n_brightness_bins"]
    all_patches: dict[int, list[np.ndarray]] = {i: [] for i in range(n_bins)}

    for path in image_paths:
        print(f"  Processing: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"    WARN: Cannot read {path}, skipping.")
            continue

        patches = extract_noise_patches(img, config)
        for bin_idx, plist in patches.items():
            all_patches[bin_idx].extend(plist)
            print(f"    Bin {bin_idx}: +{len(plist)} patches")

    # 彙整
    db = {"config": config}
    total = 0
    for bin_idx in range(n_bins):
        plist = all_patches[bin_idx]
        if len(plist) > 0:
            db[f"patches_bin_{bin_idx}"] = np.stack(plist, axis=0)
        else:
            db[f"patches_bin_{bin_idx}"] = np.zeros(
                (0, config["patch_size"], config["patch_size"], 3),
                dtype=np.float32
            )
        total += len(plist)
        print(f"  Bin {bin_idx}: {len(plist)} patches total")

    print(f"\n  Total patches: {total}")

    # 統計
    db["stats"] = {
        "total_patches": total,
        "per_bin": {i: len(all_patches[i]) for i in range(n_bins)},
        "n_images": len(image_paths),
    }

    return db


# ============================================================
# Core: Noise Synthesis
# ============================================================

def create_blend_mask(patch_size: int, margin: int) -> np.ndarray:
    """
    建立 feathered blending mask（cosine taper）。
    中心為 1.0，邊緣 margin 像素內從 0 漸變到 1。
    """
    mask_1d = np.ones(patch_size, dtype=np.float32)
    if margin > 0:
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(margin) / margin))
        mask_1d[:margin] = taper
        mask_1d[-margin:] = taper[::-1]

    mask_2d = np.outer(mask_1d, mask_1d)
    return mask_2d


def augment_patch(
    patch: np.ndarray,
    allow_flip: bool = True,
    allow_rotate: bool = True,
    intensity_jitter: float = 0.1
) -> np.ndarray:
    """
    隨機增強噪聲 patch：翻轉、旋轉、振幅抖動。
    """
    p = patch.copy()

    # 隨機翻轉
    if allow_flip:
        if np.random.random() < 0.5:
            p = p[::-1, :, ...]   # 垂直翻轉
        if np.random.random() < 0.5:
            p = p[:, ::-1, ...]   # 水平翻轉

    # 隨機 90° 旋轉
    if allow_rotate:
        k = np.random.randint(0, 4)
        if k > 0:
            p = np.rot90(p, k, axes=(0, 1))

    # 振幅抖動
    if intensity_jitter > 0:
        jitter = 1.0 + np.random.uniform(-intensity_jitter, intensity_jitter)
        p = p * jitter

    return np.ascontiguousarray(p)


def synthesize_noise(
    clean_img: np.ndarray,
    db: dict,
    config: dict
) -> np.ndarray:
    """
    將噪聲 patch 合成到乾淨影像的平坦區域。

    流程：
    1. 偵測乾淨影像的平坦區域
    2. 對每個平坦 patch 位置，根據亮度從對應 bin 隨機取噪聲 patch
    3. 以 feathered blending 貼上

    Returns:
        noisy_img: uint8, 合成後的噪聲影像
    """
    ps = config["patch_size"]
    margin = config["blend_margin"]
    n_bins = config["n_brightness_bins"]
    step = ps // 2  # 50% overlap for synthesis ensures full coverage

    h, w = clean_img.shape[:2]

    # 偵測平坦區域
    flat_mask = detect_flat_mask(clean_img, config["flat_var_threshold"])

    # 亮度圖
    if len(clean_img.shape) == 3:
        brightness = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    else:
        brightness = clean_img.copy()

    bin_edges = np.linspace(0, 256, n_bins + 1)

    # Blending buffers
    noise_accum = np.zeros_like(clean_img, dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)
    blend_mask = create_blend_mask(ps, margin)

    patches_applied = 0

    for y in range(0, h - ps + 1, step):
        for x in range(0, w - ps + 1, step):
            # 檢查是否為平坦區域
            patch_flat = flat_mask[y:y + ps, x:x + ps]
            if np.mean(patch_flat) < config["min_flat_ratio"]:
                continue

            # 計算亮度 bin
            patch_brightness = np.mean(brightness[y:y + ps, x:x + ps])
            bin_idx = int(np.clip(
                np.searchsorted(bin_edges[1:], patch_brightness),
                0, n_bins - 1
            ))

            # 從資料庫取噪聲 patch
            bin_key = f"patches_bin_{bin_idx}"
            bin_patches = db.get(bin_key)
            if bin_patches is None or len(bin_patches) == 0:
                # Fallback: 嘗試相鄰 bin
                for offset in [1, -1, 2, -2]:
                    alt_idx = np.clip(bin_idx + offset, 0, n_bins - 1)
                    alt_key = f"patches_bin_{alt_idx}"
                    alt_patches = db.get(alt_key)
                    if alt_patches is not None and len(alt_patches) > 0:
                        bin_patches = alt_patches
                        break
                if bin_patches is None or len(bin_patches) == 0:
                    continue

            # 隨機選取並增強
            idx = np.random.randint(0, len(bin_patches))
            noise_patch = augment_patch(
                bin_patches[idx],
                allow_flip=config["allow_flip"],
                allow_rotate=config["allow_rotate"],
                intensity_jitter=config["intensity_jitter"]
            )

            # 確保 patch 尺寸匹配
            if noise_patch.shape[0] != ps or noise_patch.shape[1] != ps:
                continue

            # Feathered blending 累積
            if len(clean_img.shape) == 3:
                for c in range(clean_img.shape[2]):
                    noise_accum[y:y + ps, x:x + ps, c] += (
                        noise_patch[:, :, c] * blend_mask
                        if len(noise_patch.shape) == 3
                        else noise_patch * blend_mask
                    )
            else:
                noise_accum[y:y + ps, x:x + ps] += noise_patch * blend_mask

            weight_accum[y:y + ps, x:x + ps] += blend_mask
            patches_applied += 1

    # 正規化加權噪聲
    mask = weight_accum > 0
    if len(clean_img.shape) == 3:
        for c in range(clean_img.shape[2]):
            noise_accum[:, :, c][mask] /= weight_accum[mask]
    else:
        noise_accum[mask] /= weight_accum[mask]

    # 合成：clean + noise, clamp to [0, 255]
    result = clean_img.astype(np.float64) + noise_accum
    result = np.clip(result, 0, 255).astype(np.uint8)

    print(f"    Applied {patches_applied} noise patches")
    return result


# ============================================================
# CLI: Extract
# ============================================================

def cmd_extract(args):
    """從噪聲影像建立噪聲 patch 資料庫。"""
    config = DEFAULT_CONFIG.copy()
    if args.patch_size:
        config["patch_size"] = args.patch_size
    if args.median_kernel:
        config["median_kernel"] = args.median_kernel
    if args.flat_threshold:
        config["flat_var_threshold"] = args.flat_threshold

    # 收集影像路徑
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in exts
        )
    else:
        print(f"ERROR: {args.input} is not a valid file or directory.")
        sys.exit(1)

    if not image_paths:
        print("ERROR: No images found.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print(f"Config: patch_size={config['patch_size']}, "
          f"median_kernel={config['median_kernel']}, "
          f"flat_threshold={config['flat_var_threshold']}")

    # 自動閾值（如果未指定）
    if args.auto_threshold:
        print("Auto-detecting flat threshold...")
        thresholds = []
        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                t = auto_flat_threshold(img, percentile=args.auto_percentile)
                thresholds.append(t)
                print(f"  {p.name}: auto threshold = {t:.1f}")
        if thresholds:
            config["flat_var_threshold"] = float(np.median(thresholds))
            print(f"  Using median threshold: {config['flat_var_threshold']:.1f}")

    # 建立資料庫
    print("\nExtracting noise patches...")
    db = build_noise_database(image_paths, config)

    # 儲存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 將 config 和 stats 轉為可序列化格式
    save_dict = {}
    for key, val in db.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, dict):
            # config 和 stats 存為 JSON string
            import json
            save_dict[key] = np.array(json.dumps(val, ensure_ascii=False))
        else:
            save_dict[key] = np.array(val)

    np.savez_compressed(str(output_path), **save_dict)
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved noise database to: {output_path} ({file_size:.1f} MB)")


# ============================================================
# CLI: Synthesize
# ============================================================

def cmd_synthesize(args):
    """將噪聲資料庫套用到乾淨影像。"""
    import json

    # 載入資料庫
    print(f"Loading noise database: {args.db}")
    data = np.load(args.db, allow_pickle=True)

    db = {}
    config = DEFAULT_CONFIG.copy()
    for key in data.files:
        if key == "config":
            loaded_config = json.loads(str(data[key]))
            config.update(loaded_config)
        elif key == "stats":
            stats = json.loads(str(data[key]))
            print(f"  Database stats: {stats['total_patches']} patches "
                  f"from {stats['n_images']} image(s)")
        else:
            db[key] = data[key]

    # Override config
    if args.intensity_jitter is not None:
        config["intensity_jitter"] = args.intensity_jitter
    if args.no_rotate:
        config["allow_rotate"] = False
    if args.no_flip:
        config["allow_flip"] = False

    # 收集乾淨影像
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in exts
        )
    else:
        print(f"ERROR: {args.input} is not a valid file or directory.")
        sys.exit(1)

    # 輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合成
    n_variants = args.n_variants
    print(f"\nSynthesizing {n_variants} variant(s) per image...")

    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}")
        clean_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if clean_img is None:
            print(f"    WARN: Cannot read, skipping.")
            continue

        for v in range(n_variants):
            noisy = synthesize_noise(clean_img, db, config)
            suffix = f"_noisy_{v:03d}" if n_variants > 1 else "_noisy"
            out_name = f"{img_path.stem}{suffix}{img_path.suffix}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), noisy)
            print(f"    Saved: {out_name}")

    print(f"\nDone. Output in: {output_dir}")


# ============================================================
# CLI: Visualize
# ============================================================

def cmd_visualize(args):
    """視覺化噪聲資料庫內容。"""
    import json

    data = np.load(args.db, allow_pickle=True)
    config = DEFAULT_CONFIG.copy()

    for key in data.files:
        if key == "config":
            config.update(json.loads(str(data[key])))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bins = config["n_brightness_bins"]
    ps = config["patch_size"]

    for bin_idx in range(n_bins):
        key = f"patches_bin_{bin_idx}"
        if key not in data.files:
            continue
        patches = data[key]
        if len(patches) == 0:
            print(f"  Bin {bin_idx}: empty")
            continue

        n_show = min(len(patches), 16)  # 最多顯示 16 個
        cols = min(n_show, 4)
        rows = (n_show + cols - 1) // cols

        # 建立 grid 圖
        grid_h = rows * ps + (rows - 1) * 2
        grid_w = cols * ps + (cols - 1) * 2
        # 噪聲 residual 可能有負值，做 offset 以顯示
        grid = np.full((grid_h, grid_w, 3), 128, dtype=np.uint8)

        for i in range(n_show):
            r, c = divmod(i, cols)
            y = r * (ps + 2)
            x = c * (ps + 2)
            # 將 float32 residual 映射到 [0, 255] 以視覺化
            # 128 = 零點，>128 = 正殘差，<128 = 負殘差
            vis = np.clip(patches[i] + 128, 0, 255).astype(np.uint8)
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            grid[y:y + ps, x:x + ps] = vis

        out_path = output_dir / f"noise_bin_{bin_idx}_n{len(patches)}.png"
        cv2.imwrite(str(out_path), grid)
        print(f"  Bin {bin_idx}: {len(patches)} patches → {out_path.name}")

        # 統計圖
        flat_patches = patches.reshape(len(patches), -1)
        mean_abs = np.mean(np.abs(flat_patches), axis=1)
        print(f"    Mean |residual|: {np.mean(mean_abs):.2f} "
              f"(min={np.min(mean_abs):.2f}, max={np.max(mean_abs):.2f})")

    print(f"\nVisualization saved to: {output_dir}")


# ============================================================
# CLI: Analyze
# ============================================================

def cmd_analyze(args):
    """分析噪聲影像的噪聲統計特性，幫助設定參數。"""
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in exts
        )
    else:
        print(f"ERROR: {args.input} is not a valid file or directory.")
        sys.exit(1)

    for kernel in [11, 15, 21, 31, 41]:
        print(f"\n--- Median kernel = {kernel} ---")
        for img_path in image_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            residual, pseudo_clean = extract_noise_residual(img, kernel)

            abs_res = np.abs(residual)
            print(f"  {img_path.name}:")
            print(f"    Residual stats: "
                  f"mean={np.mean(abs_res):.2f}, "
                  f"std={np.std(residual):.2f}, "
                  f"max={np.max(abs_res):.1f}")
            print(f"    Non-zero ratio: "
                  f"{np.mean(abs_res > 1.0):.1%} (>1), "
                  f"{np.mean(abs_res > 5.0):.1%} (>5), "
                  f"{np.mean(abs_res > 10.0):.1%} (>10)")

            # 自動閾值建議
            threshold = auto_flat_threshold(img)
            flat_mask = detect_flat_mask(img, threshold)
            print(f"    Auto flat threshold: {threshold:.1f} "
                  f"(covers {np.mean(flat_mask):.1%} of image)")

    print("\n建議：")
    print("  1. 選擇 residual mean 穩定時的最小 kernel 作為 --median-kernel")
    print("  2. 若 non-zero ratio (>1) 很高，表示 kernel 太小，仍有噪聲殘留")
    print("  3. auto flat threshold 可用 --auto-threshold 自動套用")


# ============================================================
# CLI: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise Patch Sampler: 從少量噪聲影像抽取噪聲 patch 並合成到乾淨影像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 分析噪聲特性（選擇最佳參數）
  python noise_patch_sampler.py analyze -i noisy_images/

  # 抽取噪聲 patch
  python noise_patch_sampler.py extract -i noisy_images/ -o noise_db.npz --auto-threshold

  # 合成噪聲影像（每張乾淨圖生成 5 個變體）
  python noise_patch_sampler.py synthesize -i clean_images/ -d noise_db.npz -o output/ -n 5

  # 視覺化噪聲資料庫
  python noise_patch_sampler.py visualize -d noise_db.npz -o viz/
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="操作模式")

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="從噪聲影像抽取噪聲 patch")
    p_extract.add_argument("-i", "--input", required=True,
                           help="噪聲影像路徑（檔案或目錄）")
    p_extract.add_argument("-o", "--output", default="noise_db.npz",
                           help="輸出資料庫路徑 (default: noise_db.npz)")
    p_extract.add_argument("--patch-size", type=int, default=None,
                           help=f"Patch 邊長 (default: {DEFAULT_CONFIG['patch_size']})")
    p_extract.add_argument("--median-kernel", type=int, default=None,
                           help=f"Median filter kernel (default: {DEFAULT_CONFIG['median_kernel']})")
    p_extract.add_argument("--flat-threshold", type=float, default=None,
                           help=f"平坦區域方差閾值 (default: {DEFAULT_CONFIG['flat_var_threshold']})")
    p_extract.add_argument("--auto-threshold", action="store_true",
                           help="自動偵測平坦閾值")
    p_extract.add_argument("--auto-percentile", type=float, default=30.0,
                           help="自動閾值的百分位數 (default: 30.0)")

    # --- synthesize ---
    p_synth = subparsers.add_parser("synthesize", help="將噪聲合成到乾淨影像")
    p_synth.add_argument("-i", "--input", required=True,
                         help="乾淨影像路徑（檔案或目錄）")
    p_synth.add_argument("-d", "--db", required=True,
                         help="噪聲資料庫路徑 (.npz)")
    p_synth.add_argument("-o", "--output", default="output/",
                         help="輸出目錄 (default: output/)")
    p_synth.add_argument("-n", "--n-variants", type=int, default=1,
                         help="每張乾淨圖生成幾個變體 (default: 1)")
    p_synth.add_argument("--intensity-jitter", type=float, default=None,
                         help="噪聲振幅抖動比例 (default: 0.1)")
    p_synth.add_argument("--no-rotate", action="store_true",
                         help="禁止 90° 旋轉增強（噪聲具嚴格方向性時使用）")
    p_synth.add_argument("--no-flip", action="store_true",
                         help="禁止翻轉增強")

    # --- visualize ---
    p_viz = subparsers.add_parser("visualize", help="視覺化噪聲資料庫")
    p_viz.add_argument("-d", "--db", required=True,
                       help="噪聲資料庫路徑 (.npz)")
    p_viz.add_argument("-o", "--output", default="viz/",
                       help="輸出目錄 (default: viz/)")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="分析噪聲特性")
    p_analyze.add_argument("-i", "--input", required=True,
                           help="噪聲影像路徑（檔案或目錄）")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print(f"{'=' * 50}")
    print(f"Noise Patch Sampler — {args.command}")
    print(f"{'=' * 50}\n")

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "synthesize":
        cmd_synthesize(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
