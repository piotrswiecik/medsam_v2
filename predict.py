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
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import MedSAMSyntax
from typedefs import Config
from preprocessing import apply_clahe

# CLAHE preprocessing configuration
# TODO: Move to config
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)


def run_prediction(config: Config, image_path: str, checkpoint_path: str):
    """
    Run inference on a single image using trained MedSAMSyntax model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    sam_model = sam_model.to(device)

    print(f"[INFO] SAM ViT-B model loaded on {device}")
    print(
        f"[INFO] Total parameters: {sum(p.numel() for p in sam_model.parameters()):,}"
    )

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

    # Apply CLAHE preprocessing (same as training)
    # TODO: Move to config
    if USE_CLAHE:
        print(
            f"[INFO] Applying CLAHE preprocessing (clip_limit={CLAHE_CLIP_LIMIT}, tile_size={CLAHE_TILE_GRID_SIZE})"
        )
        image = apply_clahe(image, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE)

    # Run YOLO to get bbox (same as training)
    print(f"[INFO] Loading YOLO model: {config.yolo_model_path}")
    yolo_model = YOLO(str(config.yolo_model_path))

    print(f"[INFO] Running YOLO inference...")
    results = yolo_model.predict(
        source=str(image_path),
        imgsz=config.image_size,
        conf=config.yolo_confidence_threshold,
        verbose=False,
    )

    # Extract combined bbox
    if (
        not results
        or len(results) == 0
        or results[0].boxes is None
        or len(results[0].boxes) == 0
    ):
        print(f"   [WARNING] No YOLO detections, using full image bbox")
        bbox = np.array([[0, 0, 1024, 1024]], dtype=np.float32)
        num_detections = 0
        avg_conf = 0.0
    else:
        boxes = results[0].boxes
        all_coords = boxes.xyxy.cpu().numpy()
        all_confs = boxes.conf.cpu().numpy()

        x_min = int(all_coords[:, 0].min())
        y_min = int(all_coords[:, 1].min())
        x_max = int(all_coords[:, 2].max())
        y_max = int(all_coords[:, 3].max())

        bbox = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
        num_detections = len(boxes)
        avg_conf = float(all_confs.mean())

        print(
            f"   [INFO] YOLO detected {num_detections} objects (avg conf: {avg_conf:.3f})"
        )
        print(f"   [INFO] Combined bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")

    # Normalize using the SAME approach as training
    # Training uses: A.Normalize(mean=(0.5,), std=(0.5,))
    # This maps [0, 255] -> [0, 1] -> [-1, 1]
    transform = A.Compose(
        [
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )

    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Create model
    model = MedSAMSyntax(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        num_classes=26,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Model loaded (epoch {checkpoint['epoch']+1})")
    if "val_miou" in checkpoint:
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
    axes[0].imshow(
        cv2.cvtColor(cv2.resize(original_image, (1024, 1024)), cv2.COLOR_BGR2RGB)
    )
    axes[0].set_title(f"Original Image\n{image_path.name}")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=25)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    # Overlay
    img_for_overlay = cv2.cvtColor(
        cv2.resize(original_image, (1024, 1024)), cv2.COLOR_BGR2RGB
    )
    axes[2].imshow(img_for_overlay)
    axes[2].imshow(pred_mask, cmap="tab20", alpha=0.5, vmin=0, vmax=25)
    axes[2].set_title("Overlay (Prediction + Image)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # save prediction image
    if not os.path.exists(os.path.join(config.medsam_workdir, "pred")):
        os.makedirs(os.path.join(config.medsam_workdir, "pred"))
    plt.savefig(
        os.path.join(config.medsam_workdir, "pred", f"{image_path.stem}_prediction.png")
    )
    print(
        f"Prediction visualization saved to: {os.path.join(config.medsam_workdir, 'pred', f'{image_path.stem}_prediction.png')}"
    )

    return pred_mask
