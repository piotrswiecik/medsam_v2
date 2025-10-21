from ultralytics import YOLO
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from pycocotools.coco import COCO
import torch
import numpy as np
import urllib.request
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from segment_anything import sam_model_registry

from train import MedSAMSyntax
from typedefs import Config


def run_prediction(config: Config, image_path: str, checkpoint_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load base SAM model (without weights first, we'll load trained weights later)
    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    sam_model = sam_model.to(device)

    print(f"[INFO] SAM ViT-B model loaded on {device}")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in sam_model.parameters()):,}")

    print(f"[INFO] MedSAMSyntax model created")
    print(f"   [INFO] Num classes: 26")
    print(f"   [INFO] Device: {device}")

    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"[INFO] Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    print(f"   [INFO] Original shape: {original_shape}")
    
    # Resize to 1024x1024
    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image_normalized = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Full image bbox (fallback if no YOLO available)
    bbox = np.array([[0, 0, 1024, 1024]], dtype=np.float32)
    
    print(f"🔧 Loading checkpoint: {Path(checkpoint_path).name}")

    # Create model
    model = MedSAMSyntax(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        num_classes=26
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']+1})")
    if 'val_miou' in checkpoint:
        print(f"   Validation mIoU: {checkpoint['val_miou']:.3f}")
    
    # Run inference
    print(f"🔮 Running inference...")
    with torch.no_grad():
        semantic_logits, iou_pred = model(image_tensor, bbox)
    
    # Get prediction
    pred_mask = torch.argmax(semantic_logits[0], dim=0).cpu().numpy()
    
    print(f"✅ Inference complete")
    print(f"   Unique classes predicted: {np.unique(pred_mask)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(cv2.resize(original_image, (1024, 1024)), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image\n{image_path.name}')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_mask, cmap='tab20', vmin=0, vmax=25)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')
    
    # Overlay
    img_for_overlay = cv2.cvtColor(cv2.resize(original_image, (1024, 1024)), cv2.COLOR_BGR2RGB)
    axes[2].imshow(img_for_overlay)
    axes[2].imshow(pred_mask, cmap='tab20', alpha=0.5, vmin=0, vmax=25)
    axes[2].set_title('Overlay (Prediction + Image)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_mask