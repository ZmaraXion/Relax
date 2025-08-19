"""  
Required packages:  
- torch  
- torchvision  
- opencv-python  
- numpy  
- tqdm  
- argparse  
- PIL (Pillow)  
"""  
  
import argparse  
import os  
import torch  
import cv2  
import numpy as np  
from PIL import Image  
from torchvision import transforms  
from tqdm import tqdm  
import torch.nn.functional as F  
  
  
def create_transform():  
    """Create preprocessing transform for SSCD model inference."""  
    # Based on SSCD recommended preprocessing  
    normalize = transforms.Normalize(  
        mean=[0.485, 0.456, 0.406],   
        std=[0.229, 0.224, 0.225]  
    )  
    return transforms.Compose([  
        transforms.Resize([320, 320]),  # Square resize for efficiency  
        transforms.ToTensor(),  
        normalize,  
    ])  
  
  
def load_model(model_path, device):  
    """Load SSCD TorchScript model."""  
    print(f"Loading model from: {model_path}")  
    model = torch.jit.load(model_path, map_location=device)  
    model.eval()  
    return model  
  
  
def load_query_frames(query_dir, transform):  
    """Load and preprocess 5 query frames."""  
    query_frames = []  
    frame_files = sorted([f for f in os.listdir(query_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])  
      
    if len(frame_files) != 5:  
        raise ValueError(f"Query directory must contain exactly 5 frames, found {len(frame_files)}")  
      
    print(f"Loading {len(frame_files)} query frames...")  
    for frame_file in frame_files:  
        frame_path = os.path.join(query_dir, frame_file)  
        img = Image.open(frame_path).convert('RGB')  
        img_tensor = transform(img)  
        query_frames.append(img_tensor)  
      
    return torch.stack(query_frames)  
  
  
def extract_video_embeddings(video_path, model, transform, batch_size, device):  
    """Extract embeddings from all video frames using batched processing."""  
    cap = cv2.VideoCapture(video_path)  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
      
    if total_frames == 0:  
        raise ValueError(f"Could not read video or video is empty: {video_path}")  
      
    print(f"Processing {total_frames} frames from video...")  
      
    embeddings = []  
    batch_frames = []  
      
    with tqdm(total=total_frames, desc="Extracting embeddings") as pbar:  
        for frame_idx in range(total_frames):  
            ret, frame = cap.read()  
            if not ret:  
                break  
                  
            # Convert BGR to RGB  
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            img = Image.fromarray(frame_rgb)  
            img_tensor = transform(img)  
            batch_frames.append(img_tensor)  
              
            # Process batch when full or at the end  
            if len(batch_frames) == batch_size or frame_idx == total_frames - 1:  
                batch_tensor = torch.stack(batch_frames).to(device)  
                  
                with torch.no_grad():  
                    batch_embeddings = model(batch_tensor)  
                  
                embeddings.append(batch_embeddings.cpu())  
                batch_frames = []  
                pbar.update(len(batch_tensor))  
      
    cap.release()  
      
    # Concatenate all embeddings  
    all_embeddings = torch.cat(embeddings, dim=0)  
    print(f"Extracted embeddings shape: {all_embeddings.shape}")  
      
    return all_embeddings  
  
  
def compute_query_embedding(query_frames, model, device):  
    """Compute single representative embedding for query frames."""  
    print("Computing query embedding...")  
    query_frames = query_frames.to(device)  
      
    with torch.no_grad():  
        query_embeddings = model(query_frames)  
      
    # Average the 5 query embeddings to get single representative vector  
    query_vector = torch.mean(query_embeddings, dim=0, keepdim=True)  
    return query_vector.cpu()  
  
  
def find_best_sequence(query_vector, video_embeddings):  
    """Find the best matching 5-frame sequence using sliding window."""  
    print("Searching for best matching sequence...")  
      
    num_frames = video_embeddings.shape[0]  
    if num_frames < 5:  
        raise ValueError(f"Video must have at least 5 frames, got {num_frames}")  
      
    best_similarity = -1.0  
    best_start_frame = 0  
      
    # Sliding window of size 5  
    for start_idx in range(num_frames - 4):  
        # Get 5-frame window  
        window_embeddings = video_embeddings[start_idx:start_idx + 5]  
          
        # Compute average embedding for this window  
        window_vector = torch.mean(window_embeddings, dim=0, keepdim=True)  
          
        # Compute cosine similarity  
        similarity = F.cosine_similarity(query_vector, window_vector).item()  
          
        if similarity > best_similarity:  
            best_similarity = similarity  
            best_start_frame = start_idx  
      
    return best_start_frame, best_similarity  
  
  
def main():  
    parser = argparse.ArgumentParser(description="Video Frame Retrieval using SSCD")  
    parser.add_argument("--video_path", required=True, help="Path to reference MP4 video")  
    parser.add_argument("--query_dir", required=True, help="Directory containing 5 query frames")  
    parser.add_argument("--model_path", required=True, help="Path to SSCD TorchScript model (.pt file)")  
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")  
      
    args = parser.parse_args()  
      
    # Validate inputs  
    if not os.path.exists(args.video_path):  
        raise FileNotFoundError(f"Video file not found: {args.video_path}")  
    if not os.path.exists(args.query_dir):  
        raise FileNotFoundError(f"Query directory not found: {args.query_dir}")  
    if not os.path.exists(args.model_path):  
        raise FileNotFoundError(f"Model file not found: {args.model_path}")  
      
    # Setup device  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Using device: {device}")  
      
    # Create preprocessing transform  
    transform = create_transform()  
      
    # Load model  
    model = load_model(args.model_path, device)  
      
    # Load query frames  
    query_frames = load_query_frames(args.query_dir, transform)  
      
    # Compute query embedding  
    query_vector = compute_query_embedding(query_frames, model, device)  
      
    # Extract video embeddings  
    video_embeddings = extract_video_embeddings(  
        args.video_path, model, transform, args.batch_size, device  
    )  
      
    # Find best matching sequence  
    best_start_frame, best_similarity = find_best_sequence(query_vector, video_embeddings)  
      
    # Output result  
    print("\n" + "="*60)  
    print(f"✅ Best match found. The 5-frame sequence starts at frame: {best_start_frame} (Similarity: {best_similarity:.3f})")  
    print("="*60)  
  
  
if __name__ == "__main__":  
    main()
