#!/usr/bin/env python3
"""
Video Frame Retrieval Demo using Pre-trained K400 Model
Identifies the most similar sequence of 5 consecutive frames in a target video
that matches a given query of 5 consecutive frames.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.io import read_video
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
import argparse
from easydict import EasyDict
from copy import deepcopy
import torchvision.models as models

# ============================================================================
# Model Architecture (Based on CARL TransformerModel)
# ============================================================================

def attention(Q, K, V, mask=None, dropout=None):
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)
    
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    
    if dropout is not None:
        out = dropout(out)
    
    return out

class MultiheadedAttention(nn.Module):
    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None, d_out=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_out = d_out
        if self.d_out is None:
            self.d_out = self.d_model_Q
        if self.d_model is None:
            self.d_model = self.d_model_Q
        
        self.d_k = self.d_model // H
        
        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_out)
        
        self.dropout = nn.Dropout(self.dout_p)
        
        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask=None):
        B, Sq, d_model_Q = Q.shape
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)
        
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        Q = attention(Q, K, V, mask, self.dropout)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        Q = self.linear_d2Q(Q)
        
        return Q

def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def generate_sincos_embedding(seq_len, d_model, train_len=None):
    odds = np.arange(0, d_model, 2)
    evens = np.arange(1, d_model, 2)
    pos_enc_mat = np.zeros((seq_len, d_model))
    if train_len is None:
        pos_list = np.arange(seq_len)
    else:
        pos_list = np.linspace(0, train_len-1, num=seq_len)

    for i, pos in enumerate(pos_list):
        pos_enc_mat[i, odds] = np.sin(pos / (10000 ** (odds / d_model)))
        pos_enc_mat[i, evens] = np.cos(pos / (10000 ** (evens / d_model)))

    return torch.from_numpy(pos_enc_mat).unsqueeze(0)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dout_p, seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        self.seq_len = seq_len

    def forward(self, x):
        B, S, d_model = x.shape
        if S != self.seq_len:
            pos_enc_mat = generate_sincos_embedding(S, d_model, self.seq_len)
            x = x + pos_enc_mat.type_as(x)
        else:
            pos_enc_mat = generate_sincos_embedding(S, d_model)
            x = x + pos_enc_mat.type_as(x)
        x = self.dropout(x)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        return x + res

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dout_p = dout_p
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dout_p, H=8, d_ff=None, d_hidden=None):
        super(EncoderLayer, self).__init__()
        self.res_layer0 = ResidualConnection(d_model, dout_p)
        self.res_layer1 = ResidualConnection(d_model, dout_p)
        if d_hidden is None: 
            d_hidden = d_model
        if d_ff is None: 
            d_ff = 4*d_model
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H, d_model=d_hidden)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, src_mask=None):
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layer0(x, sublayer0)
        x = self.res_layer1(x, sublayer1)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, dout_p, H, d_ff, N, d_hidden=None):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff, d_hidden), N)
        
    def forward(self, x, src_mask=None):
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

class TransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        self.pooling = nn.AdaptiveMaxPool2d(1)
        
        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.BatchNorm1d(channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        self.video_emb = nn.Linear(in_channels, hidden_channels)
        
        self.video_pos_enc = PositionalEncoder(hidden_channels, drop_rate, seq_len=cfg.TRAIN.NUM_FRAMES)
        if cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            self.video_encoder = Encoder(hidden_channels, drop_rate, cfg.MODEL.EMBEDDER_MODEL.NUM_HEADS, 
                                        cfg.MODEL.EMBEDDER_MODEL.D_FF, cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)
        
        self.embedding_layer = nn.Linear(hidden_channels, self.embedding_size)

    def forward(self, x, video_masks=None):
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)

        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.view(batch_size, num_steps, x.size(1))
        x = self.video_pos_enc(x)
        if self.cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS > 0:
            x = self.video_encoder(x, src_mask=video_masks)

        x = x.view(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x

class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        res50_model = models.resnet50(pretrained=True)
        if cfg.MODEL.BASE_MODEL.LAYER == 3:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-3])
            self.res_finetune = list(res50_model.children())[-3]
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
        elif cfg.MODEL.BASE_MODEL.LAYER == 2:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-4])
            self.res_finetune = nn.Sequential(*list(res50_model.children())[-4:-2])
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
        else:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-2])
            self.res_finetune = nn.Identity()
            cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
        self.embed = TransformerEmbModel(cfg)
        self.embedding_size = self.embed.embedding_size

    def forward(self, x, num_frames=None, video_masks=None, project=False, classification=False):
        batch_size, num_steps, c, h, w = x.shape
        frames_per_batch = self.cfg.MODEL.BASE_MODEL.FRAMES_PER_BATCH
        num_blocks = int(math.ceil(float(num_steps)/frames_per_batch))
        backbone_out = []
        for i in range(num_blocks):
            curr_idx = i * frames_per_batch
            cur_steps = min(num_steps-curr_idx, frames_per_batch)
            curr_data = x[:, curr_idx:curr_idx+cur_steps]
            curr_data = curr_data.contiguous().view(-1, c, h, w)
            self.backbone.eval()
            with torch.no_grad():
                curr_emb = self.backbone(curr_data)
            curr_emb = self.res_finetune(curr_emb)
            _, out_c, out_h, out_w = curr_emb.size()
            curr_emb = curr_emb.contiguous().view(batch_size, cur_steps, out_c, out_h, out_w)
            backbone_out.append(curr_emb)
        x = torch.cat(backbone_out, dim=1)
        
        x = self.embed(x, video_masks=video_masks)

        if self.cfg.MODEL.L2_NORMALIZE:
            x = F.normalize(x, dim=-1)
        return x

# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def resize_frames(images, size):
    """Resize frames to target size."""
    return torch.nn.functional.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

def uniform_crop(images, size, spatial_idx=1):
    """Perform uniform spatial cropping on images."""
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped

def color_normalization(images, mean=[0.485, 0.456, 0.406], stddev=[0.229, 0.224, 0.225]):
    """Apply ImageNet color normalization."""
    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
    return out_images

def preprocess_frames(frames, image_size=224):
    """
    Preprocess frames for model input.
    Applies uniform crop, resize, and color normalization.
    """
    # Uniform crop to square
    frames = uniform_crop(frames, min(frames.shape[2], frames.shape[3]))
    # Resize to target size
    frames = resize_frames(frames, image_size)
    # Color normalization (ImageNet stats)
    frames = color_normalization(frames)
    return frames

# ============================================================================
# Model Configuration and Loading
# ============================================================================

def create_k400_config():
    """Create configuration for K400 pre-trained model."""
    cfg = EasyDict()
    
    # Basic settings
    cfg.IMAGE_SIZE = 224
    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_FRAMES = 80
    
    # Model configuration
    cfg.MODEL = EasyDict()
    cfg.MODEL.BASE_MODEL = EasyDict()
    cfg.MODEL.BASE_MODEL.LAYER = 3
    cfg.MODEL.BASE_MODEL.NETWORK = 'Resnet50_byol'
    cfg.MODEL.BASE_MODEL.FRAMES_PER_BATCH = 40
    cfg.MODEL.BASE_MODEL.OUT_CHANNEL = 2048
    
    cfg.MODEL.EMBEDDER_MODEL = EasyDict()
    cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR = 2
    cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE = 256
    cfg.MODEL.EMBEDDER_MODEL.D_FF = 1024
    cfg.MODEL.EMBEDDER_MODEL.NUM_HEADS = 8
    cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS = 2
    cfg.MODEL.EMBEDDER_MODEL.CONV_LAYERS = [[256, 3, 1], [256, 3, 1]]
    cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE = 128
    cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
    cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS = [[256, True], [256, True]]
    cfg.MODEL.EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
    cfg.MODEL.EMBEDDER_MODEL.USE_BN = True
    
    cfg.MODEL.EMBEDDER_TYPE = 'transformer'
    cfg.MODEL.L2_NORMALIZE = True
    cfg.MODEL.PROJECTION = True
    cfg.MODEL.PROJECTION_HIDDEN_SIZE = 512
    cfg.MODEL.PROJECTION_SIZE = 128
    cfg.MODEL.TRAIN_BASE = 'frozen'
    
    return cfg

def load_model_checkpoint(model_path, device='cuda'):
    """
    Load pre-trained model from checkpoint.
    """
    # Create model configuration
    cfg = create_k400_config()
    
    # Initialize model
    model = TransformerModel(cfg)
    
    # Load checkpoint
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {model_path}")
    
    model.to(device)
    model.eval()
    return model, cfg

# ============================================================================
# Video Processing Functions
# ============================================================================

def load_video_frames(video_path):
    """
    Load video frames using torchvision.
    Returns frames as tensor with shape (T, C, H, W) in [0, 1] range.
    """
    print(f"Loading video: {video_path}")
    video, _, info = read_video(video_path, pts_unit='sec')
    
    if len(video) == 0:
        raise ValueError(f"Could not load video from {video_path}")
    
    # Convert from (T, H, W, C) to (T, C, H, W) and normalize to [0, 1]
    video = video.permute(0, 3, 1, 2).float() / 255.0
    print(f"Loaded video with {len(video)} frames, resolution: {video.shape[2]}x{video.shape[3]}")
    
    return video

def load_query_frames(query_paths):
    """
    Load query frames from image files.
    Returns frames as tensor with shape (5, C, H, W) in [0, 1] range.
    """
    frames = []
    for path in sorted(query_paths):
        print(f"Loading query frame: {path}")
        # Load using PIL and convert to tensor
        img = Image.open(path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        frames.append(img_tensor)
    
    query_frames = torch.stack(frames)
    print(f"Loaded {len(query_frames)} query frames")
    return query_frames

def create_sliding_windows(video_frames, window_size=5):
    """
    Create sliding windows of consecutive frames.
    Returns list of frame windows and their starting indices.
    """
    num_frames = len(video_frames)
    if num_frames < window_size:
        print(f"Warning: Video has only {num_frames} frames, less than window size {window_size}")
        return [video_frames], [0]
    
    windows = []
    start_indices = []
    
    for i in range(num_frames - window_size + 1):
        window = video_frames[i:i + window_size]
        windows.append(window)
        start_indices.append(i)
    
    print(f"Created {len(windows)} sliding windows of size {window_size}")
    return windows, start_indices

def extract_embeddings(model, cfg, frames, device='cuda'):
    """
    Extract embeddings from frames using the model.
    """
    with torch.no_grad():
        # Add batch dimension and move to device
        if len(frames.shape) == 4:  # (T, C, H, W)
            frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        
        frames = frames.to(device)
        
        # Preprocess frames
        batch_size, num_steps, c, h, w = frames.shape
        frames_flat = frames.view(batch_size * num_steps, c, h, w)
        frames_preprocessed = preprocess_frames(frames_flat, cfg.IMAGE_SIZE)
        frames_preprocessed = frames_preprocessed.view(batch_size, num_steps, c, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        
        # Get embeddings
        embeddings = model(frames_preprocessed, num_frames=num_steps)
        
        # Average embeddings across time dimension for each sequence
        embeddings = embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        
    return embeddings.cpu().numpy()

# ============================================================================
# Similarity Computation and Retrieval
# ============================================================================

def compute_cosine_similarity(query_embedding, reference_embeddings):
    """
    Compute cosine similarity between query and reference embeddings.
    Returns similarity scores (higher = more similar).
    """
    # Use scipy's cdist with cosine distance, then convert to similarity
    distances = cdist(query_embedding.reshape(1, -1), reference_embeddings, metric='cosine')[0]
    similarities = 1 - distances  # Convert distance to similarity
    return similarities

def find_best_match(similarities):
    """
    Find the index of the best matching window.
    """
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    return best_idx, best_score

# ============================================================================
# Main Demo Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Video Frame Retrieval Demo')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pre-trained K400 model (.pth or .ckpt file)')
    parser.add_argument('--video_path', type=str, default='video1.mp4',
                        help='Path to reference video file')
    parser.add_argument('--query_frames', type=str, nargs=5, 
                        default=['query_frame_1.jpg', 'query_frame_2.jpg', 'query_frame_3.jpg', 
                                'query_frame_4.jpg', 'query_frame_5.jpg'],
                        help='Paths to 5 consecutive query frames')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Video Frame Retrieval Demo using Pre-trained K400 Model")
    print("=" * 80)
    
    # Load the pre-trained model
    print("\n1. Loading pre-trained model...")
    model, cfg = load_model_checkpoint(args.model_path, args.device)
    
    # Load reference video
    print("\n2. Processing reference video...")
    video_frames = load_video_frames(args.video_path)
    
    # Create sliding windows of 5 consecutive frames
    print("\n3. Creating sliding windows...")
    windows, start_indices = create_sliding_windows(video_frames, window_size=5)
    
    # Extract embeddings for all windows
    print("\n4. Extracting embeddings from reference video windows...")
    reference_embeddings = []
    for i, window in enumerate(windows):
        if i % 100 == 0:  # Progress indicator
            print(f"   Processing window {i+1}/{len(windows)}")
        
        embedding = extract_embeddings(model, cfg, window, args.device)
        reference_embeddings.append(embedding[0])  # Remove batch dimension
    
    reference_embeddings = np.array(reference_embeddings)
    print(f"   Extracted embeddings for {len(reference_embeddings)} windows")
    
    # Load and process query frames
    print("\n5. Processing query frames...")
    query_frames = load_query_frames(args.query_frames)
    
    # Extract embedding for query
    print("\n6. Extracting query embedding...")
    query_embedding = extract_embeddings(model, cfg, query_frames, args.device)[0]
    
    # Compute similarities
    print("\n7. Computing similarities...")
    similarities = compute_cosine_similarity(query_embedding, reference_embeddings)
    
    # Find best match
    print("\n8. Finding best match...")
    best_idx, best_score = find_best_match(similarities)
    best_start_frame = start_indices[best_idx]
    
    # Output results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Query matched best at frame index: {best_start_frame} in {args.video_path}")
    print(f"Similarity score: {best_score:.4f}")
    print(f"Best matching window: frames {best_start_frame} to {best_start_frame + 4}")
    
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"- Total reference windows processed: {len(reference_embeddings)}")
    print(f"- Average similarity score: {similarities.mean():.4f}")
    print(f"- Similarity standard deviation: {similarities.std():.4f}")
    print(f"- Top 5 matching frame indices: {np.argsort(similarities)[-5:][::-1].tolist()}")

if __name__ == "__main__":
    main()
```

This script implements a complete video frame retrieval system based on the CARL codebase architecture. Here are the key technical decisions supported by the codebase:

**Model Architecture**: The script uses the TransformerModel architecture which combines a ResNet50 backbone with a Transformer-based embedder [1](#0-0) .

**Model Configuration**: The K400 configuration is based on the pre-training setup, including 128-dimensional embeddings, transformer with 2 layers, 8 heads, and L2 normalization [2](#0-1) .

**Data Preprocessing**: The preprocessing pipeline follows the evaluation setup without data augmentation, using uniform crop, resize to 224x224, and ImageNet color normalization [3](#0-2) .

**Video Loading**: Video frames are loaded using torchvision.io.read_video and converted to float tensors in [0,1] range [4](#0-3) .

**Sliding Window Sampling**: The frame sampling strategy creates sliding windows similar to the evaluation approach used in the codebase [5](#0-4) .

**Similarity Computation**: Cosine similarity is computed using scipy's cdist function, which is the same approach used in the retrieval evaluation [6](#0-5) .

**Notes**:
- The script requires PyTorch, torchvision, scipy, PIL, and numpy
- Make sure to install the correct versions compatible with your system
- The model expects input frames preprocessed with ImageNet normalization
- The script handles both .pth and .ckpt checkpoint formats
- GPU acceleration is used when available for faster processing
