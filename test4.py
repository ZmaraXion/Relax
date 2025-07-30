import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Tuple, Optional

def load_video_frames(video_path: str, resize: Tuple[int, int] = (160, 160)) -> List[np.ndarray]:
    """Load video frames, resize, and convert to grayscale for optical flow.
    
    Args:
        video_path: Path to the video file.
        resize: Target resolution for frames (width, height).
    
    Returns:
        List of preprocessed grayscale frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and convert to grayscale
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    
    cap.release()
    if len(frames) < 5:
        raise ValueError(f"Video {video_path} has too few frames (< 5)")
    
    return frames

def compute_optical_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """Compute Farneback optical flow between two grayscale frames.
    
    Args:
        prev_frame: Previous grayscale frame.
        next_frame: Next grayscale frame.
    
    Returns:
        Optical flow field (height, width, 2).
    """
    # Apply Gaussian blur to reduce noise
    prev_frame = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    next_frame = cv2.GaussianBlur(next_frame, (5, 5), 0)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, next_frame, None,
        pyr_scale=0.5, levels=2, winsize=13,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    return flow

def compute_motion_similarity(flows1: List[np.ndarray], flows2: List[np.ndarray], 
                            min_magnitude: float = 0.1) -> float:
    """Compute motion similarity based on optical flow direction.
    
    Args:
        flows1: List of optical flow fields for first group.
        flows2: List of optical flow fields for second group.
        min_magnitude: Minimum flow magnitude to filter noise.
    
    Returns:
        Motion similarity score (0 to 1).
    """
    if len(flows1) != len(flows2):
        return 0.0
    
    direction_similarities = []
    for flow1, flow2 in zip(flows1, flows2):
        # Flatten flow fields
        flow1 = flow1.reshape(-1, 2)
        flow2 = flow2.reshape(-1, 2)
        
        # Filter small magnitude vectors
        mag1 = np.sqrt(np.sum(flow1**2, axis=1))
        mag2 = np.sqrt(np.sum(flow2**2, axis=1))
        valid_mask = (mag1 > min_magnitude) & (mag2 > min_magnitude)
        flow1 = flow1[valid_mask]
        flow2 = flow2[valid_mask]
        
        if len(flow1) < 10:  # Too few valid vectors
            return 0.0
        
        # Compute cosine similarity for direction
        dot_product = np.sum(flow1 * flow2, axis=1)
        norm1 = np.sqrt(np.sum(flow1**2, axis=1))
        norm2 = np.sqrt(np.sum(flow2**2, axis=1))
        cos_sim = np.mean(dot_product / (norm1 * norm2 + 1e-8))
        direction_similarities.append(cos_sim)
    
    return float(np.mean(direction_similarities))

def load_sscd_model(model_path: str = None) -> torch.nn.Module:
    """Load SSCCD model (placeholder for actual model).
    
    Args:
        model_path: Path to pretrained SSCCD model (optional).
    
    Returns:
        PyTorch model for SSCCD.
    """
    # Placeholder: replace with actual model loading
    model = torch.hub.load('facebookresearch/sscd-copy-detection', 'resnet50', pretrained=True)
    model.eval()
    return model.to('cpu')

def compute_sscd_similarity(group1: List[np.ndarray], group2: List[np.ndarray], 
                          model: torch.nn.Module) -> Tuple[float, float]:
    """Compute SSCCD similarity for two groups of frames using batch processing.
    
    Args:
        group1: List of frames from first video.
        group2: List of frames from second video.
        model: SSCCD model for descriptor extraction.
    
    Returns:
        Tuple of (mean similarity, standard deviation of similarities).
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert frames to batch
    batch1 = torch.stack([transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) 
                         for frame in group1])
    batch2 = torch.stack([transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) 
                         for frame in group2])
    
    # Compute descriptors
    with torch.no_grad():
        desc1 = model(batch1)
        desc2 = model(batch2)
    
    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(desc1, desc2).cpu().numpy()
    
    return float(np.mean(similarities)), float(np.std(similarities))

def align_videos(video1_path: str, video2_path: str, model: torch.nn.Module,
                 frame_group_size: int = 5, sim_threshold: float = 0.9,
                 std_threshold: float = 0.05, motion_threshold: float = 0.8,
                 slide_step: int = 2) -> Optional[int]:
    """Align two videos by finding the best matching frame group.
    
    Args:
        video1_path: Path to first video.
        video2_path: Path to second video.
        model: SSCCD model for similarity computation.
        frame_group_size: Number of frames in each group (default: 5).
        sim_threshold: Minimum SSCCD similarity threshold.
        std_threshold: Maximum SSCCD similarity standard deviation.
        motion_threshold: Minimum motion similarity threshold.
        slide_step: Step size for sliding window in video2.
    
    Returns:
        Best matching frame index in video2, or None if no match found.
    """
    try:
        # Load and preprocess frames
        color_frames1 = load_video_frames(video1_path, resize=(160, 160))
        color_frames2 = load_video_frames(video2_path, resize=(160, 160))
        gray_frames1 = load_video_frames(video1_path, resize=(160, 160))
        gray_frames2 = load_video_frames(video2_path, resize=(160, 160))
        
        # Randomly select a group from video1
        start_idx1 = np.random.randint(0, len(color_frames1) - frame_group_size + 1)
        group1_color = color_frames1[start_idx1:start_idx1 + frame_group_size]
        group1_gray = gray_frames1[start_idx1:start_idx1 + frame_group_size]
        
        best_score = -float('inf')
        best_index = None
        
        # Slide over video2 with step size
        for i in range(0, len(color_frames2) - frame_group_size + 1, slide_step):
            group2_color = color_frames2[i:i + frame_group_size]
            group2_gray = gray_frames2[i:i + frame_group_size]
            
            # Compute SSCCD similarity
            mean_sim, std_sim = compute_sscd_similarity(group1_color, group2_color, model)
            
            if mean_sim < sim_threshold or std_sim > std_threshold:
                continue
            
            # Compute optical flows
            flows1 = [compute_optical_flow(group1_gray[j], group1_gray[j+1]) 
                     for j in range(frame_group_size-1)]
            flows2 = [compute_optical_flow(group2_gray[j], group2_gray[j+1]) 
                     for j in range(frame_group_size-1)]
            
            # Compute motion similarity
            motion_sim = compute_motion_similarity(flows1, flows2)
            
            if motion_sim < motion_threshold:
                continue
            
            # Combine scores
            combined_score = 0.7 * mean_sim + 0.3 * motion_sim
            
            if combined_score > best_score:
                best_score = combined_score
                best_index = i
        
        return best_index
    
    except Exception as e:
        print(f"Error during video alignment: {e}")
        return None

def main():
    """Main function to demonstrate video alignment."""
    try:
        video1_path = "path/to/video1.mp4"
        video2_path = "path/to/video2.mp4"
        model = load_sscd_model()
        
        best_index = align_videos(video1_path, video2_path, model)
        if best_index is not None:
            print(f"Best matching frame index in video2: {best_index}")
        else:
            print("No matching frame group found.")
    
    except Exception as e:
        print(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
