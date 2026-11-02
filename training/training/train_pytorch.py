"""
PyTorch GPU Training Pipeline for Plant Disease Classification.
Leverages NVIDIA GTX 1650 CUDA 12.4 with AMP (Automatic Mixed Precision).
Features:
- Live terminal progress bar with ETA, loss, and accuracy.
- Automatic Mixed Precision (AMP) to maximize speed on GTX 1650.
- Decoupled weight decay (AdamW) with Cosine Annealing learning rate schedule.
- Label smoothing cross-entropy.
- Model checkpointing & early stopping.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.pytorch_model import OptimizedPlantCNN, build_pytorch_mobilenet_v3

def get_dataloaders(train_dir: str, val_dir: str, batch_size: int = 32):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading image datasets from disk...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"  ├── Train Images: {len(train_dataset):,} across {len(train_dataset.classes)} classes")
    print(f"  └── Valid Images: {len(val_dataset):,} across {len(val_dataset.classes)} classes")

    # num_workers=0 avoids Windows spawn overhead
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset.classes

def format_time(seconds: float) -> str:
    """Formats seconds into readable mm:ss or hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def train_pytorch(
    train_dir: str,
    val_dir: str,
    model_name: str = "custom_cnn",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_str: str = "cuda"
):
    print("=" * 65)
    print("🌿 PLANT DISEASE DEEP LEARNING GPU TRAINING PIPELINE")
    print("=" * 65)

    # Device selection
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 Device: {gpu_name} ({total_vram:.1f} GB VRAM) | CUDA {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("ℹ️ Device: CPU")

    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader, classes = get_dataloaders(train_dir, val_dir, batch_size=batch_size)
    num_classes = len(classes)
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)

    print(f"Batch Size: {batch_size} | Batches per Epoch: {total_train_batches:,}")

    # Model
    if model_name == "mobilenet_v3":
        model = build_pytorch_mobilenet_v3(num_classes=num_classes)
    else:
        model = OptimizedPlantCNN(num_classes=num_classes)
    model.to(device)

    # Modern AMP & Optimizer (eliminates PyTorch 2.6 deprecation warnings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    print("=" * 65)
    print("Starting Training Loop...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # ------------------ TRAINING PHASE ------------------
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            total_train += labels.size(0)

            # Live Progress Bar in Terminal every 25 batches
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_train_batches:
                elapsed = time.time() - epoch_start_time
                batches_done = batch_idx + 1
                rate = batches_done / (elapsed + 1e-6)
                remaining_batches = total_train_batches - batches_done
                eta = remaining_batches / (rate + 1e-6)

                bar_len = 20
                filled = int(bar_len * (batches_done / total_train_batches))
                bar = "█" * filled + "░" * (bar_len - filled)

                current_loss = train_loss / total_train
                current_acc = (train_correct / total_train) * 100

                vram_str = ""
                if use_cuda:
                    mem_mb = torch.cuda.memory_allocated() / (1024**2)
                    vram_str = f" | VRAM: {mem_mb:.0f}MB"

                print(
                    f"\rEpoch {epoch:02d}/{epochs:02d} [{bar}] {batches_done}/{total_train_batches} "
                    f"({(batches_done/total_train_batches)*100:4.1f}%) | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:5.1f}% | "
                    f"ETA: {format_time(eta)}{vram_str}",
                    end="",
                    flush=True
                )

        scheduler.step()
        print()  # Newline after training phase

        # ------------------ VALIDATION PHASE ------------------
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                total_val += labels.size(0)

        epoch_duration = time.time() - epoch_start_time
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct / total_train) * 100
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct / total_val) * 100

        print(
            f"✅ Epoch {epoch:02d}/{epochs:02d} Summary ({format_time(epoch_duration)}):\n"
            f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%\n"
            f"   Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%"
        )

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            save_path = f"checkpoints/best_{model_name}_pytorch.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   🏆 New best validation loss! Saved model to {save_path}")
        else:
            patience_counter += 1
            print(f"   (No improvement for {patience_counter}/{patience} epochs)")
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered at Epoch {epoch} to prevent overfitting.")
                break
        print("-" * 65)

    print("\n🎉 Training Complete! You can now run the app:")
    print("   streamlit run main.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_pytorch(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
