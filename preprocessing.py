import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse


def apply_clahe(image, clip_limit, tile_grid_size):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    if len(image.shape) == 3:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb
    else:
        return enhanced


def explore_clahe_parameters(image_path: str):
    """
    Test different CLAHE parameter combinations and visualize results.
    
    Parameter grid:
    - clip_limit: [1.0, 2.0, 3.0, 4.0] - controls contrast enhancement strength
    - tile_grid_size: [(8,8), (16,16)] - controls locality of enhancement
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"[INFO] Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_shape = image.shape[:2]
    print(f"[INFO] Original shape: {original_shape}")
    
    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        print(f"[INFO] Resized to: 1024x1024")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    clip_limits = [1.0, 2.0, 3.0, 4.0]  
    tile_sizes = [(8, 8), (16, 16)]   
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image\n(No CLAHE)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    idx = 1
    results = []
    
    for tile_size in tile_sizes:
        for clip_limit in clip_limits:
            print(f"[INFO] Processing: clip_limit={clip_limit}, tile_size={tile_size}")
            
            enhanced = apply_clahe(image, clip_limit, tile_size)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(enhanced_rgb)
            axes[idx].set_title(
                f'CLAHE\nclip={clip_limit}, tile={tile_size}',
                fontsize=11
            )
            axes[idx].axis('off')
            
            results.append({
                'image': enhanced,
                'clip_limit': clip_limit,
                'tile_size': tile_size
            })
            
            idx += 1
    
    plt.tight_layout()
    
    output_dir = Path("data/clahe_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{image_path.stem}_clahe_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved visualization: {output_file}")
    
    for i, result in enumerate(results):
        filename = output_dir / f"{image_path.stem}_clip{result['clip_limit']}_tile{result['tile_size'][0]}x{result['tile_size'][1]}.png"
        cv2.imwrite(str(filename), result['image'])
    
    print(f"[INFO] Saved {len(results)} individual CLAHE variants to: {output_dir}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Explore CLAHE parameters for coronary angiography preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image file'
    )
    
    args = parser.parse_args()
    
    try:
        explore_clahe_parameters(args.image_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
