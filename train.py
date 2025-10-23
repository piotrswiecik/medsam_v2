"""Integrated training pipeline for YOLO+MedSAM"""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import gdown
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typedefs import Config

SPLITS = ["train", "val", "test"]

AUGMENTATION_PARAMS = {
    "HORIZONTAL_FLIP_P": 0.5,
    "VERTICAL_FLIP_P": 0.5,
    "RANDOM_ROTATE_90_P": 0.5,
    "ROTATE_LIMIT": 15,
    "ROTATE_P": 0.3,
    "ELASTIC_TRANSFORM_ALPHA": 1.0,
    "ELASTIC_TRANSFORM_SIGMA": 50.0,
    "ELASTIC_TRANSFORM_P": 0.2,
    "GRID_DISTORTION_P": 0.2,
    "OPTICAL_DISTORTION_LIMIT": 0,
    "OPTICAL_DISTORTION_P": 0.2,
    "RANDOM_BRC_B_LIMIT": 0.1,
    "RANDOM_BRC_C_LIMIT": 0.1,
    "RANDOM_BRC_P": 0.3,
    "GAUSS_NOISE_P": 0.2,
    "BLUR_LIMIT": 5,
    "BLUR_P": 0.2,
}


class MetricsTracker:
    """Track IoU and other metrics for semantic segmentation"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred_logits, targets):
        """Update metrics with batch predictions"""
        pred = torch.argmax(pred_logits, dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        pred = pred.flatten()
        targets = targets.flatten()
        
        for t, p in zip(targets, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def get_metrics(self):
        """Calculate metrics from confusion matrix"""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        accuracy = np.sum(tp) / np.sum(self.confusion_matrix)
        mean_iou = np.mean(iou[1:])  # Skip background
        mean_precision = np.mean(precision[1:])
        mean_recall = np.mean(recall[1:])
        mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'per_class_iou': iou
        }


class MedSAMSyntax(nn.Module):
    """MedSAM adapted for SYNTAX 26-class semantic segmentation

    Key fix: Uses rich decoder features instead of final binary mask for classification.
    """

    def __init__(self, image_encoder, mask_decoder, prompt_encoder, num_classes=26):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.num_classes = num_classes

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        decoder_feature_dim = 32

        self.semantic_head = nn.Sequential(
            nn.ConvTranspose2d(decoder_feature_dim, 128, kernel_size=4, stride=4),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def _extract_decoder_features(self, image_embedding, image_pe, sparse_embeddings, dense_embeddings):
        """
        Extract features from SAM mask decoder.

        Returns:
            upscaled_features: (1, 32, 64, 64)
            iou_predictions: (1, 1)
        """
        decoder = self.mask_decoder

        output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0) # [5, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1) # [B, 5, 256]
        tokens = torch.cat((output_tokens, sparse_embeddings), dim=1) # [B, 7, 256] (2 sparse + 5 output)

        src = torch.repeat_interleave(image_embedding, tokens.shape[0], dim=0)
        src = src + dense_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = decoder.output_upscaling(src)  # (1, 32, 64, 64)

        iou_pred = decoder.iou_prediction_head(iou_token_out)  # (1, num_mask_tokens)

        iou_pred = iou_pred[:, 0:1]

        return upscaled_embedding, iou_pred

    def forward(self, image, boxes):
        """
        Args:
            image: (B, 3, 1024, 1024) input images
            boxes: (B, 4) bounding boxes [x_min, y_min, x_max, y_max] as numpy array

        Returns:
            semantic_logits: (B, num_classes, 1024, 1024)
            iou_predictions: (B, 1)
        """
        batch_size = image.shape[0]

        image_embedding = self.image_encoder(image) # [B, 256, 64, 64]

        if isinstance(boxes, np.ndarray):
            boxes_torch = torch.as_tensor(boxes, dtype=torch.float32, device=image.device)
        else:
            boxes_torch = boxes

        all_semantic_logits = []
        all_iou_predictions = []

        for i in range(batch_size):
            single_image_embedding = image_embedding[i:i+1]
            single_box = boxes_torch[i:i+1]

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=single_box,
                    masks=None
                ) # ([B, 2, 256], [B, 256, 64, 64])
                image_pe = self.prompt_encoder.get_dense_pe() # [B, 256, 64, 64] TODO verify

            upscaled_features, iou_predictions = self._extract_decoder_features(
                single_image_embedding,
                image_pe,
                sparse_embeddings,
                dense_embeddings
            )

            # upscaled_features: (1, 32, 64, 64) -> semantic_logits: (1, 26, 256, 256)
            semantic_logits_256 = self.semantic_head(upscaled_features)

            # Upsample to full resolution (1024x1024)
            semantic_logits = F.interpolate(
                semantic_logits_256,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            all_semantic_logits.append(semantic_logits)
            all_iou_predictions.append(iou_predictions)

        # Concatenate results
        final_semantic_logits = torch.cat(all_semantic_logits, dim=0)
        final_iou_predictions = torch.cat(all_iou_predictions, dim=0)

        return final_semantic_logits, final_iou_predictions


def compute_class_weights(class_pixel_counts, num_classes):
    """
    Compute normalized inverse frequency weights for class imbalance.
    
    Args:
        class_pixel_counts: dict mapping class_id -> pixel_count
        num_classes: total number of classes
    
    Returns:
        torch.Tensor of shape (num_classes,) with class weights
    """
    total_pixels = sum(class_pixel_counts.values())
    weights = torch.zeros(num_classes)
    
    for cls_id in range(num_classes):
        count = class_pixel_counts.get(cls_id, 1)  # Avoid division by zero
        weights[cls_id] = total_pixels / (num_classes * count)
    
    # Normalize so weights sum to num_classes
    weights = weights * num_classes / weights.sum()
    
    return weights


class CombinedLoss(nn.Module):
    """Combined Weighted CE + Dice Loss for semantic segmentation"""
    
    def __init__(self, num_classes, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def dice_loss(self, inputs, targets, smooth=1e-5):
        """Multi-class Dice loss"""
        inputs = F.softmax(inputs, dim=1)
        
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for c in range(1, self.num_classes):
            input_c = inputs[:, c].flatten()
            target_c = targets_one_hot[:, c].flatten()
            
            intersection = (input_c * target_c).sum()
            dice = (2. * intersection + smooth) / (input_c.sum() + target_c.sum() + smooth)
            dice_scores.append(dice)
        
        dice_score = torch.stack(dice_scores).mean()
        return 1 - dice_score
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes, H, W) logits
            targets: (B, H, W) class indices
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        ce_loss = F.cross_entropy(
            inputs, targets.long(),
            weight=self.class_weights
        )
        
        dice_loss = self.dice_loss(inputs, targets)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss, {
            'ce': ce_loss.item(),
            'dice': dice_loss.item(),
            'total': total_loss.item()
        }


class SyntaxDataset:
    def __init__(self, data_root, split, bbox_cache_path, img_size=1024, original_size=512, use_augmentations=True):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.original_size = original_size
        self.scale_factor = img_size / original_size
        self.use_augmentations = use_augmentations
        self.augmentation_params = AUGMENTATION_PARAMS # Fixed for testing
        
        ann_path = self.data_root / split / 'annotations' / f'{split}.json'
        self.coco = COCO(str(ann_path))
        self.image_ids = sorted(self.coco.getImgIds())
        
        with open(bbox_cache_path) as f:
            cache_data = json.load(f)
            self.bbox_cache = cache_data['bboxes']

        if self.use_augmentations:
            self.augment = A.Compose(
                [
                    A.HorizontalFlip(p=self.augmentation_params["HORIZONTAL_FLIP_P"]),
                    A.VerticalFlip(p=self.augmentation_params["VERTICAL_FLIP_P"]),
                    A.RandomRotate90(p=self.augmentation_params["RANDOM_ROTATE_90_P"]),
                    A.Rotate(
                        limit=self.augmentation_params["ROTATE_LIMIT"],
                        p=self.augmentation_params["ROTATE_P"],
                    ),
                    A.ElasticTransform(
                        alpha=self.augmentation_params["ELASTIC_TRANSFORM_ALPHA"],
                        sigma=self.augmentation_params["ELASTIC_TRANSFORM_SIGMA"],
                        p=self.augmentation_params["ELASTIC_TRANSFORM_P"],
                    ),
                    A.GridDistortion(p=self.augmentation_params["GRID_DISTORTION_P"]),
                    A.OpticalDistortion(
                        distort_limit=self.augmentation_params["OPTICAL_DISTORTION_LIMIT"],
                        p=self.augmentation_params["OPTICAL_DISTORTION_P"],
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=self.augmentation_params["RANDOM_BRC_B_LIMIT"],
                        contrast_limit=self.augmentation_params["RANDOM_BRC_C_LIMIT"],
                        p=self.augmentation_params["RANDOM_BRC_P"],
                    ),
                    A.GaussNoise(p=self.augmentation_params["GAUSS_NOISE_P"]),
                    A.Blur(
                        blur_limit=self.augmentation_params["BLUR_LIMIT"],
                        p=self.augmentation_params["BLUR_P"],
                    ),
                    A.Normalize(mean=(0.5,), std=(0.5,)),
                    ToTensorV2(),
                ],
                additional_targets={"mask": "mask"},
            )
        else:
            self.augment = A.Compose(
                [
                    A.Normalize(mean=(0.5,), std=(0.5,)),
                    ToTensorV2(),
                ],
                additional_targets={"mask": "mask"},
            )
        
        print(f"[INFO] Loaded {split} split: {len(self.image_ids)} images")
        print(f"[INFO] Bbox scaling: {original_size} → {img_size} (×{self.scale_factor})")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor | int | str]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = self.data_root / self.split / 'images' / img_name
        
        # Load and resize image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Load bbox and scale to match resized image
        bbox_info = self.bbox_cache[img_name]
        bbox = bbox_info['bbox']
        
        # Scale bbox coordinates from original_size to img_size
        scaled_bbox = [int(coord * self.scale_factor) for coord in bbox]
        
        # Load and resize mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for ann in anns:
            seg_mask = self.coco.annToMask(ann)
            if seg_mask.shape != (self.img_size, self.img_size):
                seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask[seg_mask > 0] = ann['category_id']
        
        # Normalize image
        # image_normalized = image.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_normalized = (image_normalized - mean) / std
        
        # Convert to tensors
        # image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # mask_tensor = torch.from_numpy(mask).long()
        # bbox_tensor = torch.tensor(scaled_bbox, dtype=torch.float32)

        augmented = self.augment(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].long()
        bbox_tensor = torch.tensor(scaled_bbox, dtype=torch.float32)
        
        return {
            'image': image_tensor,  # shape=3x1024x1024 upsampled
            'mask': mask_tensor,    # shape=1024x1024
            'bbox': bbox_tensor,    # shape=4, rescaled
            'image_id': img_id,     # int
            'image_name': img_name  # str
        }
    

def _display_class_stats(stats, class_pixel_counts):
    """Display class distribution statistics"""
    print(f"{'Class ID':<10} {'Pixel Count':<15} {'% Total':<12} {'% Foreground':<15}")
    print("-" * 55)
    for s in stats:
        if s['fg_percentage'] is None:
            fg_str = "N/A"
        else:
            fg_str = f"{s['fg_percentage']:.4f}%"
        print(f"{s['class_id']:<10} {s['pixel_count']:<15,} {s['percentage']:<12.4f}% {fg_str:<15}")
    
    # Summary statistics
    total_pixels = sum(class_pixel_counts.values())
    foreground_pixels = total_pixels - class_pixel_counts.get(0, 0)
    background_pixels = class_pixel_counts.get(0, 0)
    
    print(f"\n{'Total pixels:':<20} {total_pixels:,}")
    print(f"{'Num classes:':<20} {len(class_pixel_counts)}")
    
    print(f"\n{'Background (class 0):':<25} {background_pixels:,} ({100*background_pixels/total_pixels:.2f}%)")
    print(f"{'Foreground (all vessels):':<25} {foreground_pixels:,} ({100*foreground_pixels/total_pixels:.2f}%)")
    print(f"{'Background:Foreground ratio:':<25} {background_pixels/max(foreground_pixels,1):.2f}:1")
    
    # Find rarest class
    vessel_stats = [s for s in stats if s['class_id'] != 0]
    if vessel_stats:
        rarest = min(vessel_stats, key=lambda x: x['pixel_count'])
        most_common = max(vessel_stats, key=lambda x: x['pixel_count'])
        print(f"\n{'Rarest vessel class:':<25} {rarest['class_id']} ({rarest['fg_percentage']:.4f}% of foreground)")
        print(f"{'Most common vessel class:':<25} {most_common['class_id']} ({most_common['fg_percentage']:.4f}% of foreground)")
        print(f"{'Vessel class imbalance:':<25} {most_common['pixel_count']/max(rarest['pixel_count'],1):.2f}:1")


def analyze_class_distribution(dataset, split_name, cache_path):
    """
    Compute class pixel counts and statistics across entire dataset.
    Results are cached to disk to avoid recomputation.
    
    Args:
        dataset: Dataset to analyze
        split_name: Name of split (e.g., 'train', 'val')
        cache_path: Path to cache file (auto-generated if None)
    
    Returns:
        stats: List of dicts with class statistics
        class_pixel_counts: Dict mapping class_id -> pixel_count
    """
    # Try to load from cache
    if cache_path.exists():
        print(f"[INFO] Loading cached class distribution from {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Convert class_pixel_counts keys back to int
            class_pixel_counts = {int(k): v for k, v in cached_data['class_pixel_counts'].items()}
            stats = cached_data['stats']
            
            print(f"[INFO] Loaded from cache (generated at {cached_data['metadata']['generated_at']})\n")
            _display_class_stats(stats, class_pixel_counts)
            return stats, class_pixel_counts
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Cache file corrupted ({e.__class__.__name__}), deleting and recomputing...")
            cache_path.unlink()
    
    # Compute from scratch (either cache doesn't exist or was corrupted)
    print(f"[INFO] Analyzing {split_name} split ({len(dataset)} images)...")
    print(f"[INFO] This will be cached to: {cache_path}")
    
    class_pixel_counts = defaultdict(int)
    total_pixels = 0
    
    start_time = time.time()
    for idx in tqdm(range(len(dataset)), desc=f"Analyzing {split_name}"):
        sample = dataset[idx]
        mask = sample['mask'].numpy()
        
        unique_classes, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_pixel_counts[cls] += count
        
        total_pixels += mask.size
    
    elapsed = time.time() - start_time
    
    # Sort by class ID
    class_ids = sorted(class_pixel_counts.keys())
    
    # Calculate foreground pixels (excluding class 0)
    foreground_pixels = total_pixels - class_pixel_counts.get(0, 0)
    
    # Compute statistics
    stats = []
    for cls_id in class_ids:
        count = class_pixel_counts[cls_id]
        percentage = 100 * count / total_pixels
        
        # Percentage relative to foreground only
        if cls_id == 0:
            fg_percentage = None
        else:
            fg_percentage = 100 * count / foreground_pixels if foreground_pixels > 0 else 0.0
        
        stats.append({
            'class_id': int(cls_id),
            'pixel_count': int(count),
            'percentage': float(percentage),
            'fg_percentage': float(fg_percentage) if fg_percentage is not None else None
        })
    
    print(f"[INFO] Analysis completed in {elapsed:.2f}s")
    
    # Save to cache
    cache_data = {
        'metadata': {
            'split': split_name,
            'num_images': len(dataset),
            'total_pixels': int(total_pixels),
            'num_classes': len(class_ids),
            'generated_at': datetime.now().isoformat(),
            'computation_time_seconds': float(elapsed)
        },
        'class_pixel_counts': {int(k): int(v) for k, v in class_pixel_counts.items()},
        'stats': stats
    }
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"[INFO] Cached to: {cache_path}\n")
    
    _display_class_stats(stats, class_pixel_counts)
    return stats, class_pixel_counts


def get_combined_bbox(yolo_model, image_path, conf_threshold=0.25, img_size=1024):
    """Run YOLO inference and return single combined bbox.
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max, num_detections, avg_conf)
        If no detections: (0, 0, img_size, img_size, 0, 0.0) -> full size bbox
    """
    results = yolo_model.predict(source=str(image_path), imgsz=img_size, conf=conf_threshold, verbose=False)
    
    if not results or len(results) == 0:
        return (0, 0, img_size, img_size, 0, 0.0)
    
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return (0, 0, img_size, img_size, 0, 0.0)
    
    all_coords = boxes.xyxy.cpu().numpy()
    all_confs = boxes.conf.cpu().numpy()
    
    x_min = int(all_coords[:, 0].min())
    y_min = int(all_coords[:, 1].min())
    x_max = int(all_coords[:, 2].max())
    y_max = int(all_coords[:, 3].max())
    
    num_detections = len(boxes)
    avg_conf = float(all_confs.mean())
    
    return (x_min, y_min, x_max, y_max, num_detections, avg_conf)


def generate_bbox_cache(yolo_model_path, data_root, splits, conf_threshold, output_dir, img_size=1024):
    yolo_model = YOLO(str(yolo_model_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        print(f"[INFO] Processing {split} split...")
        
        split_dir = Path(data_root) / split / 'images'
        image_files = sorted(list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpg')))
        
        if len(image_files) == 0:
            print(f"[WARNING] No images found in {split_dir}, skipping...")
            continue
        
        bboxes = {}
        fallback_count = 0
        
        for img_path in tqdm(image_files, desc=f"{split}"):
            x_min, y_min, x_max, y_max, num_det, avg_conf = get_combined_bbox(
                yolo_model, img_path, conf_threshold, img_size
            )
            
            is_fallback = (num_det == 0)
            if is_fallback:
                fallback_count += 1
            
            bboxes[img_path.name] = {
                'bbox': [x_min, y_min, x_max, y_max],
                'num_detections': num_det,
                'avg_confidence': avg_conf,
                'fallback': is_fallback
            }
        
        output_file = output_dir / f"{split}_bboxes.json"
        cache_data = {
            'metadata': {
                'yolo_model': str(yolo_model_path),
                'conf_threshold': conf_threshold,
                'image_size': img_size,
                'generated_at': datetime.now().isoformat(),
                'total_images': len(image_files),
                'fallback_count': fallback_count,
                'fallback_rate': f"{100 * fallback_count / len(image_files):.2f}%"
            },
            'bboxes': bboxes
        }
        
        with open(output_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"[INFO] Saved to: {output_file}")
        print(f"[INFO] Fallback rate: {cache_data['metadata']['fallback_rate']}")


def run_pipeline(config: Config):
    # check if bbox prompt cache exists for training split, generate if needed
    train_cache_file = Path(config.yolo_cache_dir) / "train_bboxes.json"
    if train_cache_file.exists():
        print(f"[INFO] Using existing YOLO bbox cache: {train_cache_file}")

    else:
        print(f"[INFO] Generating YOLO bbox cache in: {config.yolo_cache_dir}")
        generate_bbox_cache(
            yolo_model_path=config.yolo_model_path,
            data_root=config.arcade_syntax_base_dir,
            splits=SPLITS,
            conf_threshold=config.yolo_confidence_threshold,
            output_dir=config.yolo_cache_dir,
            img_size=config.image_size
        )

    # initialize datasets and dataloaders
    print(f"[INFO] Initializing datasets and dataloaders...")
    train_dataset = SyntaxDataset(
        data_root=config.arcade_syntax_base_dir, 
        split="train", 
        bbox_cache_path=os.path.join(config.yolo_cache_dir, f"train_bboxes.json"), 
        img_size=config.image_size,
        original_size=512
    )

    val_dataset = SyntaxDataset(
        data_root=config.arcade_syntax_base_dir,
        split="val", 
        bbox_cache_path=os.path.join(config.yolo_cache_dir, f"val_bboxes.json"), 
        img_size=config.image_size,
        original_size=512
    )

    train_stats, train_counts = analyze_class_distribution(train_dataset, "train", cache_path=Path(config.medsam_workdir) / "class_dist_cache" / "train_class_distribution.json")

    workdir = Path(config.medsam_workdir)
    workdir.mkdir(exist_ok=True, parents=True)

    # Download checkpoint if needed
    if not os.path.exists(config.medsam_base):
        print(f"[ERROR] SAM checkpoint not found at {config.medsam_base}")
        try:
            print(f"[INFO] Downloading MedSAM checkpoint...")
            file_id = "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, config.medsam_base, quiet=False)
            print(f"[INFO] Downloaded MedSAM checkpoint to {config.medsam_base}")
        except Exception as e:
            print(f"[ERROR] Failed to download MedSAM checkpoint: {e}")
            return

    try:
        from segment_anything import sam_model_registry
        print("[INFO] segment_anything imported")
    except ImportError:
        print("[INFO] Installing segment-anything...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "git+https://github.com/facebookresearch/segment-anything.git"])
        from segment_anything import sam_model_registry
        print("[INFO] segment_anything installed and imported")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_model = sam_model_registry["vit_b"]()
    checkpoint = torch.load(config.medsam_base, map_location=device, weights_only=False)
    sam_model.load_state_dict(checkpoint)
    sam_model = sam_model.to(device)

    print(f"[INFO] SAM ViT-B model loaded on {device}")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in sam_model.parameters()):,}")

    medsam_syntax = MedSAMSyntax(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        num_classes=26
    ).to(device)

    print(f"[INFO] MedSAMSyntax model created (with improved semantic head)")
    print(f"   [INFO] Num classes: 26")
    print(f"   [INFO] Device: {device}")
    print(f"   [INFO] Semantic head parameters: {sum(p.numel() for p in medsam_syntax.semantic_head.parameters()):,}")

    print("\n[INFO] Computing class weights from training data...")
    class_weights = compute_class_weights(train_counts, num_classes=26)
    print(f"[INFO] Class weights computed")
    print(f"   [INFO] Background (class 0) weight: {class_weights[0]:.2f}")
    print(f"   [INFO] Most common vessel (class 6) weight: {class_weights[6]:.2f}")
    print(f"   [INFO] Rarest vessel (class 12) weight: {class_weights[12]:.2f}")
    print(f"   [INFO] Weight ratio (class 12 / class 6): {class_weights[12] / class_weights[6]:.2f}×")

    criterion = CombinedLoss(
        num_classes=26,
        class_weights=class_weights.to(device),
        ce_weight=0.5,
        dice_weight=0.5
    )
    print(f"\n[INFO] CombinedLoss created (0.5 CE + 0.5 Dice)")

    trainable_params = (
        list(medsam_syntax.image_encoder.parameters()) +
        list(medsam_syntax.mask_decoder.parameters()) +
        list(medsam_syntax.semantic_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-5,
        weight_decay=0.01
    )

    print(f"[INFO] Optimizer created (AdamW)")
    print(f"   [INFO] Learning rate: 1e-5")
    print(f"   [INFO] Weight decay: 0.01")
    print(f"   [INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"   [INFO] Image encoder: TRAINABLE")
    print(f"   [INFO] Mask decoder: TRAINABLE")
    print(f"   [INFO] Prompt encoder: FROZEN")
    print(f"   [INFO] Semantic head: TRAINABLE")

    print(f"[INFO] Training dataset: {len(train_dataset)} images")
    print(f"[INFO] Validation dataset: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.medsam_batch_size,
        shuffle=True,
        num_workers=config.medsam_num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.medsam_batch_size,
        shuffle=False,
        num_workers=config.medsam_num_workers,
        pin_memory=True
    )

    print(f"\n[INFO] DataLoaders created")
    print(f"   [INFO] Batch size: {config.medsam_batch_size}")
    print(f"   [INFO] Num workers: {config.medsam_num_workers}")
    print(f"   [INFO] Training batches: {len(train_loader)}")
    print(f"   [INFO] Validation batches: {len(val_loader)}")

    print(f"\n[INFO] Testing forward pass with a single batch...")

    medsam_syntax.eval()

    test_batch = next(iter(train_loader))
    test_images = test_batch['image'].to(device)
    test_masks = test_batch['mask'].to(device)
    test_bboxes = test_batch['bbox'].numpy()

    print(f"\nInput shapes:")
    print(f"  Images: {test_images.shape}")
    print(f"  Masks: {test_masks.shape}")
    print(f"  Bboxes: {test_bboxes.shape}")

    with torch.no_grad():
        semantic_logits, iou_pred = medsam_syntax(test_images, test_bboxes)

    print(f"\nOutput shapes:")
    print(f"  Semantic logits: {semantic_logits.shape}")
    print(f"  IoU predictions: {iou_pred.shape}")

    loss, loss_dict = criterion(semantic_logits, test_masks)

    print(f"\nLoss computation:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")

    pred_mask = torch.argmax(semantic_logits[0], dim=0).cpu().numpy()
    gt_mask = test_masks[0].cpu().numpy()

    print("\n[INFO] Forward pass test successful!")
    print(f"[INFO] {pred_mask.shape=}, {gt_mask.shape=}")

    checkpoint_dir = Path(config.medsam_workdir) / "checkpoints" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    NUM_EPOCHS = config.medsam_num_epochs
    VAL_FREQ = config.medsam_val_freq
    CHECKPOINT_FREQ = config.medsam_checkpoint_freq

    print("[INFO] Training configuration:")
    print(f"   [INFO] Epochs: {NUM_EPOCHS}")
    print(f"   [INFO] Validation frequency: every {VAL_FREQ} epochs")
    print(f"   [INFO] Checkpoint frequency: every {CHECKPOINT_FREQ} epochs")
    print(f"   [INFO] Checkpoint directory: {checkpoint_dir}")

    print("\n[INFO] Starting training...")

    # Training history
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    best_val_miou = 0.0

    training_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # TRAINING
        medsam_syntax.train()
        epoch_loss = 0.0
        epoch_loss_components = {'ce': 0.0, 'dice': 0.0, 'total': 0.0}
        train_metrics = MetricsTracker(num_classes=26)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            bboxes = batch['bbox'].numpy()
            
            optimizer.zero_grad()
            
            semantic_logits, iou_pred = medsam_syntax(images, bboxes)
            
            loss, loss_dict = criterion(semantic_logits, masks)
            
            loss.backward()
            optimizer.step()
            
            train_metrics.update(semantic_logits.detach(), masks)
            
            epoch_loss += loss.item()
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_dict[key]
            
            if batch_idx % 10 == 0:
                current_metrics = train_metrics.get_metrics()
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'mIoU': f"{current_metrics['mean_iou']:.3f}"
                })
        
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        for key in epoch_loss_components:
            epoch_loss_components[key] /= num_batches
        
        train_metrics_epoch = train_metrics.get_metrics()
        train_metrics_history.append(train_metrics_epoch)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"\n[INFO] Epoch {epoch+1} Training Results ({epoch_duration:.1f}s):")
        print(f"   [INFO] Loss: {avg_loss:.4f} (CE: {epoch_loss_components['ce']:.4f}, Dice: {epoch_loss_components['dice']:.4f})")
        print(f"   [INFO] Metrics: mIoU={train_metrics_epoch['mean_iou']:.3f}, mF1={train_metrics_epoch['mean_f1']:.3f}, Acc={train_metrics_epoch['accuracy']:.3f}")
        
        # VALIDATION
        if (epoch + 1) % VAL_FREQ == 0:
            print(f"\n[INFO] Running validation...")
            medsam_syntax.eval()
            val_loss = 0.0
            val_metrics = MetricsTracker(num_classes=26)
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    bboxes = batch['bbox'].numpy()
                    
                    semantic_logits, _ = medsam_syntax(images, bboxes)
                    
                    loss, _ = criterion(semantic_logits, masks)
                    val_loss += loss.item()
                    
                    val_metrics.update(semantic_logits, masks)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_metrics_epoch = val_metrics.get_metrics()
            val_metrics_history.append(val_metrics_epoch)

            print(f"   [INFO] Validation: Loss={avg_val_loss:.4f}, mIoU={val_metrics_epoch['mean_iou']:.3f}, mF1={val_metrics_epoch['mean_f1']:.3f}, Acc={val_metrics_epoch['accuracy']:.3f}")

            # Save best model
            if val_metrics_epoch['mean_iou'] > best_val_miou:
                best_val_miou = val_metrics_epoch['mean_iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': medsam_syntax.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_miou': val_metrics_epoch['mean_iou'],
                    'val_metrics': val_metrics_epoch
                }, checkpoint_dir / 'best_model.pth')
                print(f"   ✅ New best model saved! (mIoU: {best_val_miou:.3f})")
        
        # PERIODIC CHECKPOINT
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': medsam_syntax.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'train_metrics': train_metrics_epoch,
                'best_val_miou': best_val_miou
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"   [INFO] Backup checkpoint saved (epoch {epoch+1})")
        
        print()

    training_duration = time.time() - training_start_time

    print(f"\n[INFO] Training Completed!")
    print(f"[INFO] Final Results:")
    print(f"   [INFO] Training duration: {training_duration/60:.1f} minutes")
    print(f"   [INFO] Best validation mIoU: {best_val_miou:.3f}")
    print(f"   [INFO] Final training mIoU: {train_metrics_history[-1]['mean_iou']:.3f}")
    print(f"   [INFO] Models saved in: {checkpoint_dir}") 
