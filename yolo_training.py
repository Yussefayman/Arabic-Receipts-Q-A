"""
Simple YOLO11 Training Pipeline for CORU Receipt Dataset
=========================================================
Train YOLO11 to detect: Merchant, Date, Total, Item
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import wandb


def create_dataset_yaml(
    dataset_root: str = "datasets",
    output_file: str = "receipt_data.yaml"
):
    """
    Create YAML configuration file for YOLO training
    
    Args:
        dataset_root: Root directory containing train/val/test folders
        output_file: Output YAML filename
    """
    
    # Dataset configuration
    data = {
        'path': str(Path(dataset_root).absolute()),  # Absolute path to dataset root
        'train': 'train/images',  # Relative to 'path'
        'val': 'val/images',      # Relative to 'path'
        'test': 'test/images',    # Optional
        
        # Class names (0-indexed)
        'names': {
            0: 'Merchant',
            1: 'Date',
            2: 'Total',
            3: 'Item'
        },
        
        # Number of classes
        'nc': 4
    }
    
    # Save YAML file
    with open(output_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"✅ Created dataset config: {output_file}")
    return output_file


def train_receipt_detector(
    data_yaml: str = "receipt_data.yaml",
    model_size: str = 'n',  # n=nano, s=small, m=medium, l=large, x=xlarge
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    device: str = '0',  # '0' for GPU, 'cpu' for CPU
    project: str = 'runs/detect',
    name: str = 'receipt_yolo11'
):
    """
    Train YOLO11 on receipt dataset
    
    Args:
        data_yaml: Path to dataset YAML config
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size (reduce if out of memory)
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
    """
    
    print("="*60)
    print("🚀 YOLO11 Receipt Detector Training")
    print("="*60)
    print(f"Model: YOLO11{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load pretrained YOLO11 model
    model = YOLO(f'yolo11{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        
        # Optimization settings
        patience=50,          # Early stopping patience
        save=True,            # Save checkpoints
        save_period=10,       # Save every 10 epochs
        
        # Data augmentation (good for receipts)
        augment=True,
        mosaic=1.0,          # Mosaic augmentation
        mixup=0.0,           # Mixup augmentation
        degrees=10.0,        # Rotation augmentation
        translate=0.1,       # Translation augmentation
        scale=0.5,           # Scale augmentation
        flipud=0.0,          # No vertical flip (receipts are upright)
        fliplr=0.0,          # No horizontal flip (Arabic text direction)
        
        # Performance
        workers=8,           # Number of data loading workers
        verbose=True,        # Verbose output
        plots=True           # Generate training plots
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")
    print(f"Results: {project}/{name}/")
    print("="*60)
    
    return results


def validate_model(
    model_path: str,
    data_yaml: str = "receipt_data.yaml",
    imgsz: int = 640,
    device: str = '0'
):
    """
    Validate trained model on validation/test set
    
    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml: Path to dataset YAML config
        imgsz: Image size
        device: Device to use
    """
    
    print("="*60)
    print(f"🔍 Validating Model: {model_path}")
    print("="*60)
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        plots=True
    )
    
    # Print results
    print("\n📊 Validation Results:")
    print("="*60)
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print("="*60)
    
    # Per-class metrics
    print("\n📋 Per-Class Metrics:")
    print("="*60)
    class_names = ['Merchant', 'Date', 'Total', 'Item']
    for i, name in enumerate(class_names):
        if i < len(results.box.ap50):
            print(f"{name:12} - AP50: {results.box.ap50[i]:.4f}")
    print("="*60)
    
    return results


def test_inference(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    save_path: str = None
):
    """
    Test model inference on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        conf_threshold: Confidence threshold
        save_path: Path to save annotated image
    """
    
    print(f"\n🔍 Testing inference on: {image_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(
        image_path,
        conf=conf_threshold,
        verbose=False
    )
    
    # Get detections
    detections = results[0].boxes
    
    print(f"✅ Found {len(detections)} objects")
    
    # Print detections
    class_names = ['Merchant', 'Date', 'Total', 'Item']
    for i, box in enumerate(detections):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  {i+1}. {class_names[cls]}: {conf:.2f}")
    
    # Show/save result
    if save_path:
        results[0].save(filename=save_path)
        print(f"💾 Saved annotated image to: {save_path}")
    else:
        results[0].show()
    
    return results


def export_model(
    model_path: str,
    format: str = 'onnx'
):
    """
    Export trained model to different formats for deployment
    
    Args:
        model_path: Path to trained model
        format: Export format (onnx, torchscript, tflite, etc.)
    """
    
    print(f"\n📦 Exporting model to {format.upper()} format...")
    
    model = YOLO(model_path)
    
    export_path = model.export(format=format)
    
    print(f"✅ Exported to: {export_path}")
    
    return export_path


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == "__main__":
    

    wandb.login(key="00dffdbe7d0bb4a611979f58b57f4bf92e244a95")

    # Step 1: Create dataset configuration
    print("\n📝 Step 1: Creating dataset configuration...")
    data_yaml = create_dataset_yaml(
        dataset_root="datasets",
        output_file="receipt_data.yaml"
    )
    
    # Step 2: Train the model
    print("\n🏋️ Step 2: Training YOLO11...")
    train_receipt_detector(
        data_yaml=data_yaml,
        model_size='l',      # Start with nano (fastest)
        epochs=500,          # Number of training epochs
        imgsz=640,           # Image size
        batch=16,            # Batch size (reduce if out of memory)
        device='0',          # Use GPU 0 (change to 'cpu' if no GPU)
        project='runs/detect',
        name='receipt_yolo11'
    )
    
    # Step 3: Validate the model
    print("\n✅ Step 3: Validating trained model...")
    validate_model(
        model_path='runs/detect/receipt_yolo11/weights/best.pt',
        data_yaml=data_yaml
    )
    
    # Step 4: Test inference on a sample image
    print("\n🧪 Step 4: Testing inference...")
    # Uncomment to test:
    # test_inference(
    #     model_path='runs/detect/receipt_yolo11/weights/best.pt',
    #     image_path='datasets/val/images/sample_receipt.jpg',
    #     save_path='test_result.jpg'
    # )
    
    print("\n" + "="*60)
    print("🎉 Training Pipeline Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check results in: runs/detect/receipt_yolo11/")
    print("2. Best model: runs/detect/receipt_yolo11/weights/best.pt")
    print("3. Use model for inference on new receipts")
    print("="*60)