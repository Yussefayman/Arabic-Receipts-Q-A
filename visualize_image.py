import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_yolo_label(label_path: str) -> List[Tuple]:
    """
    Load YOLO format label file
    
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    
    Args:
        label_path: Path to .txt label file
        
    Returns:
        List of (class_id, center_x, center_y, width, height)
    """
    boxes = []
    
    if not Path(label_path).exists():
        print(f"⚠️  Label file not found: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((class_id, cx, cy, w, h))
    
    return boxes


def yolo_to_xyxy(box: Tuple, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (normalized center coords) to xyxy format (pixel coords)
    
    Args:
        box: (class_id, center_x, center_y, width, height) - normalized
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (x1, y1, x2, y2) in pixel coordinates
    """
    class_id, cx, cy, w, h = box
    
    # Convert from normalized to pixel coordinates
    cx_px = cx * img_width
    cy_px = cy * img_height
    w_px = w * img_width
    h_px = h * img_height
    
    # Convert from center format to corner format
    x1 = int(cx_px - w_px / 2)
    y1 = int(cy_px - h_px / 2)
    x2 = int(cx_px + w_px / 2)
    y2 = int(cy_px + h_px / 2)
    
    return x1, y1, x2, y2


def visualize_receipt(image_path: str, label_path: str, class_names: dict):
    """
    Visualize a single receipt with its bounding boxes
    
    Args:
        image_path: Path to image file
        label_path: Path to label file
        class_names: Dictionary mapping class_id to class name
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    img_height, img_width = image.shape[:2]
    
    # Load labels
    boxes = load_yolo_label(label_path)
    
    # Define colors for each class
    colors = {
        0: (0, 255, 0),      # Merchant - Green
        1: (255, 0, 0),      # Date - Blue
        2: (0, 0, 255),      # Total - Red
        3: (255, 0, 255),    # Item - Magenta
    }
    
    # Draw bounding boxes
    for box in boxes:
        class_id = box[0]
        x1, y1, x2, y2 = yolo_to_xyxy(box, img_width, img_height)
        
        # Get color for this class
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Get class name
        class_name = class_names.get(class_id, f"Class_{class_id}")
        
        # Draw label background
        label = f"{class_name}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Make sure label fits in image
        label_y = max(y1 - 10, label_size[1] + 10)
        
        cv2.rectangle(image, 
                     (x1, label_y - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, label_y), 
                     color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb, len(boxes)


def visualize_batch(
    images_dir: str = "datasets/train/images",
    labels_dir: str = "datasets/train/labels",
    num_images: int = 10,
    save_output: bool = True,
    output_dir: str = "visualizations"
):
    """
    Visualize a batch of annotated receipts
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        num_images: Number of images to visualize
        save_output: Save visualizations to files
        output_dir: Directory to save visualizations
    """
    # Class names for CORU dataset
    class_names = {
        0: 'Merchant',
        1: 'Date',
        2: 'Total',
        3: 'Item'
    }
    
    # Get all image files
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not labels_path.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Get image files
    image_files = list(images_path.glob("*.jpg")) + \
                  list(images_path.glob("*.png")) + \
                  list(images_path.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images in dataset")
    
    # Select random images
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    print(f"🎲 Selected {len(selected_images)} random images\n")
    
    # Create output directory if saving
    if save_output:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualize each image
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    stats = {'total_boxes': 0, 'class_counts': {0: 0, 1: 0, 2: 0, 3: 0}}
    
    for idx, img_path in enumerate(selected_images):
        # Get corresponding label file
        label_file = labels_path / (img_path.stem + '.txt')
        
        print(f"[{idx+1}/{len(selected_images)}] Processing: {img_path.name}")
        
        # Visualize
        result = visualize_receipt(str(img_path), str(label_file), class_names)
        
        if result is not None:
            image_rgb, num_boxes = result
            
            # Count boxes by class
            boxes = load_yolo_label(str(label_file))
            for box in boxes:
                class_id = box[0]
                if class_id in stats['class_counts']:
                    stats['class_counts'][class_id] += 1
            
            stats['total_boxes'] += num_boxes
            
            # Display in subplot
            axes[idx].imshow(image_rgb)
            axes[idx].set_title(f"{img_path.name}\n{num_boxes} boxes", fontsize=10)
            axes[idx].axis('off')
            
            # Save individual image
            if save_output:
                output_path = Path(output_dir) / f"annotated_{img_path.name}"
                cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ Found {num_boxes} bounding boxes")
        else:
            axes[idx].text(0.5, 0.5, 'Failed to load', 
                          ha='center', va='center', fontsize=12)
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(selected_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save combined visualization
    if save_output:
        combined_output = Path(output_dir) / "combined_visualization.png"
        plt.savefig(combined_output, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved combined visualization to: {combined_output}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 ANNOTATION STATISTICS")
    print("="*60)
    print(f"Total images visualized: {len(selected_images)}")
    print(f"Total bounding boxes: {stats['total_boxes']}")
    print(f"Average boxes per image: {stats['total_boxes']/len(selected_images):.1f}")
    print("\nBoxes by class:")
    for class_id, count in stats['class_counts'].items():
        class_name = class_names[class_id]
        percentage = (count / stats['total_boxes'] * 100) if stats['total_boxes'] > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    print("="*60)


def visualize_single(
    image_path: str,
    label_path: str,
    save_path: str = None
):
    """
    Visualize a single receipt with annotations
    
    Args:
        image_path: Path to image
        label_path: Path to label file
        save_path: Optional path to save annotated image
    """
    class_names = {
        0: 'Merchant',
        1: 'Date',
        2: 'Total',
        3: 'Item'
    }
    
    print(f"📄 Visualizing: {Path(image_path).name}")
    
    result = visualize_receipt(image_path, label_path, class_names)
    
    if result is not None:
        image_rgb, num_boxes = result
        print(f"✅ Found {num_boxes} bounding boxes")
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f"{Path(image_path).name} - {num_boxes} annotations")
        plt.axis('off')
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved to: {save_path}")
        
        plt.show()
        
        # Print box details
        boxes = load_yolo_label(label_path)
        print(f"\nBox details:")
        for i, box in enumerate(boxes, 1):
            class_id = box[0]
            print(f"  Box {i}: {class_names[class_id]} (class_id={class_id})")




if __name__ == "__main__":
    
    # Example 1: Visualize 10 random images from training set
    print("="*60)
    print("Visualizing CORU Training Dataset")
    print("="*60)
    
    visualize_batch(
        images_dir="datasets/train/images",
        labels_dir="datasets/train/labels",
        num_images=10,
        save_output=True,
        output_dir="visualizations"
    )
    
    # Example 2: Visualize a single specific image
    # visualize_single(
    #     image_path="datasets/train/images/receipt_001.jpg",
    #     label_path="datasets/train/labels/receipt_001.txt",
    #     save_path="single_visualization.jpg"
    # )