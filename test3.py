hare

return to facebookresearch/sscd-copy-detection
Act as an expert Python developer and ML engineer, adhering to the best software engineering practices. Context: I am working on a video frame similarity task. I have used the SSCD_disc_mixup pre-trai...
Hide full text
Act as an expert Python developer and ML engineer, adhering to the best software engineering practices. Context: I am working on a video frame similarity task. I have used the SSCD_disc_mixup pre-trained model to extract frame-level embeddings for two videos (Video A and Video B), which are saved as NumPy files. * Input Data: * video_A_embeddings.npy: with a shape of (50, 512). * video_B_embeddings.npy: with a shape of (42, 512). Primary Goal: Create a well-structured Python script (demo.py) that randomly selects a frame from Video A and finds the frame with the highest cosine similarity in Video B. Key Requirements: * Flexible Post-processing Pipeline: * The SSCD documentation suggests that post-processing steps (centering, whitening, normalization) can improve accuracy but might be optional or used in combination. * Please implement a post-processing function, e.g., postprocess(embeddings, pca_model=None, use_centering: bool = True, use_whitening: bool = True, use_l2_norm: bool = True). * This function should allow enabling or disabling each step via boolean flags. * The default behavior should apply the recommended pipeline: Centering -> Whitening -> L2 Normalization. * Correct Whitening Implementation: * When whitening is enabled, the PCA model must be fit only on a reference dataset (we'll use Video B's embeddings for this). * This fitted PCA model must then be used to transform the embeddings of both Video A and Video B. Do not fit the PCA model on Video A's data. * Similarity Calculation: * All similarity calculations must be performed on the post-processed embeddings. * Use Cosine Similarity as the metric. Deliverable: * A single, directly executable Python script named demo.py. * Code Structure: Encapsulate the core logic into functions (e.g., load_data, postprocess, find_most_similar) for readability and reusability. * Modern Python Style: Use type hints for all function signatures (parameters and return values). * Testability: The script should start by generating two dummy embedding files, allowing it to be run immediately for verification. * Clear Comments: The code must be well-commented, especially the post-processing pipeline and the PCA fit/transform logic. * Clean Output: The script's final output should clearly print: * The index of the randomly selected frame from Video A. * The index of the most similar frame found in Video B. * The corresponding cosine similarity score.
Deep
Searched across facebookresearch/sscd-copy-detection

Thought Process
Based on my analysis of the SSCD repository, I'll create a comprehensive Python script that implements the post-processing pipeline for video frame similarity. The SSCD codebase uses FAISS codecs for post-processing, particularly PCA whitening and L2 normalization. disc_eval.py:67-72 disc_eval.py:170-180

Here's the complet
#!/usr/bin/env python3  
"""  
Video Frame Similarity using SSCD Embeddings  
Implements post-processing pipeline: Centering -> Whitening -> L2 Normalization  
"""  
  
import numpy as np  
from typing import Optional, Tuple  
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
import random  
  
  
def generate_dummy_embeddings() -> None:  
    """Generate dummy embedding files for testing purposes."""  
    print("Generating dummy embedding files...")  
      
    # Set random seed for reproducibility  
    np.random.seed(42)  
      
    # Generate dummy embeddings with some structure to make similarity meaningful  
    base_embedding_a = np.random.randn(512) * 0.5  
    base_embedding_b = np.random.randn(512) * 0.5  
      
    # Video A embeddings (50 frames) - add some noise around base  
    video_a_embeddings = np.array([  
        base_embedding_a + np.random.randn(512) * 0.3 for _ in range(50)  
    ])  
      
    # Video B embeddings (42 frames) - mix of similar and different frames  
    video_b_embeddings = []  
    for i in range(42):  
        if i < 10:  # First 10 frames similar to Video A  
            video_b_embeddings.append(base_embedding_a + np.random.randn(512) * 0.2)  
        else:  # Rest are more different  
            video_b_embeddings.append(base_embedding_b + np.random.randn(512) * 0.3)  
      
    video_b_embeddings = np.array(video_b_embeddings)  
      
    # Save to files  
    np.save("video_A_embeddings.npy", video_a_embeddings)  
    np.save("video_B_embeddings.npy", video_b_embeddings)  
      
    print(f"Generated video_A_embeddings.npy: {video_a_embeddings.shape}")  
    print(f"Generated video_B_embeddings.npy: {video_b_embeddings.shape}")  
  
  
def load_data() -> Tuple[np.ndarray, np.ndarray]:  
    """  
    Load embedding data from NumPy files.  
      
    Returns:  
        Tuple of (video_A_embeddings, video_B_embeddings)  
    """  
    try:  
        video_a = np.load("video_A_embeddings.npy")  
        video_b = np.load("video_B_embeddings.npy")  
        print(f"Loaded Video A embeddings: {video_a.shape}")  
        print(f"Loaded Video B embeddings: {video_b.shape}")  
        return video_a, video_b  
    except FileNotFoundError as e:  
        print(f"Error loading files: {e}")  
        print("Generating dummy data instead...")  
        generate_dummy_embeddings()  
        return load_data()  
  
  
def postprocess(  
    embeddings: np.ndarray,  
    pca_model: Optional[PCA] = None,  
    use_centering: bool = True,  
    use_whitening: bool = True,  
    use_l2_norm: bool = True  
) -> Tuple[np.ndarray, Optional[PCA]]:  
    """  
    Apply post-processing pipeline to embeddings.  
      
    The recommended pipeline from SSCD is: Centering -> Whitening -> L2 Normalization  
      
    Args:  
        embeddings: Input embeddings of shape (n_samples, n_features)  
        pca_model: Pre-fitted PCA model for whitening. If None and use_whitening=True,  
                  a new PCA model will be fitted on the input embeddings.  
        use_centering: Whether to center the embeddings (subtract mean)  
        use_whitening: Whether to apply PCA whitening  
        use_l2_norm: Whether to apply L2 normalization  
      
    Returns:  
        Tuple of (processed_embeddings, pca_model_used)  
    """  
    processed = embeddings.copy()  
    pca_used = pca_model  
      
    # Step 1: Centering (subtract mean)  
    if use_centering:  
        if pca_model is not None:  
            # Use the mean from the reference dataset (stored in PCA model)  
            processed = processed - pca_model.mean_  
        else:  
            # Center using current dataset mean  
            processed = processed - np.mean(processed, axis=0, keepdims=True)  
        print("Applied centering")  
      
    # Step 2: PCA Whitening  
    if use_whitening:  
        if pca_model is None:  
            # Fit PCA on current embeddings (this should be the reference dataset)  
            pca_used = PCA(whiten=True)  
            processed = pca_used.fit_transform(processed)  
            print(f"Fitted and applied PCA whitening (explained variance ratio: {pca_used.explained_variance_ratio_[:5]})")  
        else:  
            # Transform using pre-fitted PCA model  
            if use_centering:  
                # PCA expects centered data, but we already centered above  
                processed = pca_used.transform(processed)  
            else:  
                # Center and transform  
                processed = pca_used.transform(processed)  
            print("Applied pre-fitted PCA whitening")  
      
    # Step 3: L2 Normalization  
    if use_l2_norm:  
        norms = np.linalg.norm(processed, axis=1, keepdims=True)  
        # Avoid division by zero  
        norms = np.where(norms == 0, 1, norms)  
        processed = processed / norms  
        print("Applied L2 normalization")  
      
    return processed, pca_used  
  
  
def compute_cosine_similarity(query_embedding: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:  
    """  
    Compute cosine similarity between a query embedding and reference embeddings.  
      
    Args:  
        query_embedding: Single embedding vector of shape (n_features,)  
        reference_embeddings: Reference embeddings of shape (n_samples, n_features)  
      
    Returns:  
        Cosine similarity scores of shape (n_samples,)  
    """  
    # Ensure query is 2D for consistent broadcasting  
    if query_embedding.ndim == 1:  
        query_embedding = query_embedding.reshape(1, -1)  
      
    # Compute dot product (cosine similarity for L2-normalized vectors)  
    similarities = np.dot(reference_embeddings, query_embedding.T).flatten()  
      
    return similarities  
  
  
def find_most_similar(  
    video_a_embeddings: np.ndarray,  
    video_b_embeddings: np.ndarray,  
    use_centering: bool = True,  
    use_whitening: bool = True,  
    use_l2_norm: bool = True  
) -> Tuple[int, int, float]:  
    """  
    Find the most similar frame between a random frame from Video A and all frames in Video B.  
      
    Args:  
        video_a_embeddings: Video A embeddings of shape (n_frames_a, n_features)  
        video_b_embeddings: Video B embeddings of shape (n_frames_b, n_features)  
        use_centering: Whether to apply centering in post-processing  
        use_whitening: Whether to apply PCA whitening in post-processing  
        use_l2_norm: Whether to apply L2 normalization in post-processing  
      
    Returns:  
        Tuple of (selected_frame_a_idx, most_similar_frame_b_idx, similarity_score)  
    """  
    print("\n" + "="*60)  
    print("POST-PROCESSING PIPELINE")  
    print("="*60)  
      
    # Step 1: Fit PCA model on Video B (reference dataset)  
    print("\n1. Fitting PCA model on Video B (reference dataset)...")  
    video_b_processed, pca_model = postprocess(  
        video_b_embeddings,  
        pca_model=None,  
        use_centering=use_centering,  
        use_whitening=use_whitening,  
        use_l2_norm=use_l2_norm  
    )  
      
    # Step 2: Apply the same PCA model to Video A  
    print("\n2. Applying fitted PCA model to Video A...")  
    video_a_processed, _ = postprocess(  
        video_a_embeddings,  
        pca_model=pca_model,  
        use_centering=use_centering,  
        use_whitening=use_whitening,  
        use_l2_norm=use_l2_norm  
    )  
      
    print("\n" + "="*60)  
    print("SIMILARITY SEARCH")  
    print("="*60)  
      
    # Step 3: Randomly select a frame from Video A  
    selected_frame_a_idx = random.randint(0, len(video_a_processed) - 1)  
    selected_embedding = video_a_processed[selected_frame_a_idx]  
      
    print(f"\nRandomly selected frame {selected_frame_a_idx} from Video A")  
      
    # Step 4: Compute similarities with all frames in Video B  
    similarities = compute_cosine_similarity(selected_embedding, video_b_processed)  
      
    # Step 5: Find the most similar frame  
    most_similar_idx = np.argmax(similarities)  
    max_similarity = similarities[most_similar_idx]  
      
    return selected_frame_a_idx, most_similar_idx, max_similarity  
  
  
def main() -> None:  
    """Main execution function."""  
    print("Video Frame Similarity using SSCD Embeddings")  
    print("=" * 50)  
      
    # Generate dummy data if files don't exist  
    generate_dummy_embeddings()  
      
    # Load embeddings  
    video_a_embeddings, video_b_embeddings = load_data()  
      
    # Set random seed for reproducible results  
    random.seed(42)  
      
    # Find most similar frame using the recommended post-processing pipeline  
    selected_a_idx, similar_b_idx, similarity_score = find_most_similar(  
        video_a_embeddings,  
        video_b_embeddings,  
        use_centering=True,    # Default: enabled  
        use_whitening=True,    # Default: enabled    
        use_l2_norm=True       # Default: enabled  
    )  
      
    # Display results  
    print("\n" + "="*60)  
    print("RESULTS")  
    print("="*60)  
    print(f"Selected frame from Video A: {selected_a_idx}")  
    print(f"Most similar frame in Video B: {similar_b_idx}")  
    print(f"Cosine similarity score: {similarity_score:.4f}")  
    print("="*60)  
  
  
if __name__ == "__main__":  
    main()
    
