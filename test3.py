# sscd_postprocessing_final.py
#
# 描述:
# 本腳本根據 "A Self-Supervised Descriptor for Image Copy Detection (SSCD)" 論文
# 及其官方 GitHub 原始碼，實現了不依賴 FAISS 的 embedding 後處理流程。
# 流程包含兩個主要步驟：
# 1. PCA 白化 (PCA Whitening)
# 2. L2 標準化 (L2 Normalization)
#
# 作者: Gemini
# 日期: 2025-07-30

import numpy as np
from sklearn.decomposition import PCA

def learn_pca_whitening(reference_embeddings: np.ndarray) -> PCA:
    """
    從參考嵌入資料集 (reference embeddings) 中學習 PCA 白化轉換。
    根據 SSCD 論文，這個轉換模型應該從一個大型的、具代表性的背景資料集
    （在此範例中，我們使用 reference_embeddings）中學習。

    Args:
        reference_embeddings (np.ndarray): 用於學習轉換的參考嵌入，維度為 (n_samples, n_features)。

    Returns:
        PCA: 一個已經 fit 好的 scikit-learn PCA 物件。
    """
    print(f"INFO: 從 {reference_embeddings.shape[0]} 個參考嵌入中學習 PCA 白化轉換...")
    
    # 根據論文 Section 5.3，SSCD 使用完整的描述符維度進行白化。
    # `whiten=True` 是 scikit-learn 中執行白化操作的關鍵參數。
    n_features = reference_embeddings.shape[1]
    pca = PCA(n_components=n_features, whiten=True)
    
    # 使用參考資料集來訓練 PCA 模型
    pca.fit(reference_embeddings)
    
    return pca

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    對一個矩陣中的每一行向量進行 L2 標準化，使其長度變為 1。

    Args:
        vectors (np.ndarray): 維度為 (n_samples, n_features) 的 numpy 矩陣。

    Returns:
        np.ndarray: 標準化後的矩陣。
    """
    # 計算每個向量的 L2 範數（長度）
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # 增加一個極小值 epsilon 來避免除以零的錯誤
    epsilon = 1e-10
    
    return vectors / (norm + epsilon)

def apply_postprocessing(embeddings: np.ndarray, pca_model: PCA) -> np.ndarray:
    """
    應用 SSCD 的後處理流程：先白化，再 L2 標準化。
    這個順序遵循論文 Section 5.3 以及官方程式碼的實現。

    Args:
        embeddings (np.ndarray): 需要進行後處理的嵌入，維度為 (n_samples, n_features)。
        pca_model (PCA): 已經從參考資料集學習好的 PCA 模型。

    Returns:
        np.ndarray: 經過完整後處理的嵌入。
    """
    # 步驟 1: 應用已學習的 PCA 白化轉換
    whitened_embeddings = pca_model.transform(embeddings)
    
    # 步驟 2: 對白化後的向量進行 L2 標準化
    final_embeddings = l2_normalize(whitened_embeddings)
    
    return final_embeddings

def main():
    """
    主執行函數，演示完整的 SSCD 後處理與相似度搜索流程。
    """
    print("--- SSCD 描述符後處理與相似度搜索展示 (不依賴 FAISS) ---")

    # 0. 準備模擬資料
    # 假設 query_embeddings 的維度是 [5, 512]
    # 假設 reference_embeddings 的維度是 [100, 512]
    descriptor_dim = 512
    num_queries = 5
    num_references = 100

    print(f"\n[步驟 0] 模擬生成 {num_queries} 個查詢 (query) 和 {num_references} 個參考 (reference) 嵌入 (維度: {descriptor_dim})...")
    # 使用 np.random.randn 生成常態分佈的數據，更接近真實世界的 embedding 分佈
    query_embeddings = np.random.randn(num_queries, descriptor_dim).astype('float32')
    reference_embeddings = np.random.randn(num_references, descriptor_dim).astype('float32')
    
    print(f"原始 Query 維度: {query_embeddings.shape}")
    print(f"原始 Reference 維度: {reference_embeddings.shape}")

    # --- 後處理流程 ---
    
    # 1. 從 Reference Embeddings 學習 PCA 白化轉換
    print("\n[步驟 1] 從 Reference Embeddings 學習 PCA 白化轉換...")
    pca_model = learn_pca_whitening(reference_embeddings)
    print("PCA 白化轉換學習完成。")

    # 2. 將學習到的轉換應用到 Query 和 Reference Embeddings
    print("\n[步驟 2] 應用後處理流程 (白化 -> L2 標準化)...")
    final_query_embeddings = apply_postprocessing(query_embeddings, pca_model)
    final_ref_embeddings = apply_postprocessing(reference_embeddings, pca_model)
    print("所有嵌入向量均已處理完成。")
    
    # 驗證 L2 標準化是否成功
    # 處理後，向量長度（L2 範數）應非常接近 1
    print(f"驗證: 處理後第一個 Query 向量的 L2 範數: {np.linalg.norm(final_query_embeddings[0]):.6f}")
    print(f"驗證: 處理後第一個 Reference 向量的 L2 範數: {np.linalg.norm(final_ref_embeddings[0]):.6f}")

    # --- 相似度搜索 ---
    print("\n[步驟 3] 開始進行相似度搜索...")
    # 因為所有向量都經過 L2 標準化，它們的內積等同於餘弦相似度。
    # 這可以透過一次矩陣乘法高效完成。
    similarity_matrix = np.dot(final_query_embeddings, final_ref_embeddings.T)
    
    print(f"計算完成的相似度矩陣維度: {similarity_matrix.shape}")
    print("(矩陣中的每個元素 (i, j) 代表第 i 個 query 和第 j 個 reference 的相似度)")
    
    # 4. 為每個 query 找到最匹配的 reference
    best_matches_indices = np.argmax(similarity_matrix, axis=1)
    best_matches_scores = np.max(similarity_matrix, axis=1)

    print("\n--- 搜索結果 ---")
    for i in range(num_queries):
        print(f"查詢 (Query) #{i}:")
        print(f"  - 最匹配的參考 (Reference) 索引: {best_matches_indices[i]}")
        print(f"  - 餘弦相似度分數: {best_matches_scores[i]:.4f}")

if __name__ == "__main__":
    main()

