import torch
import torch.nn as nn
from torch.optim import Adam
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import time

from data import make_dataloaders
from model import BeverageModelResnet

@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def train(cfg: DictConfig):

    # 1. Initialization & Config Print
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.dataset.seed)
    
    print(f"\n--- STARTING TRAINING ---")
    print(f"Device:   {device}")
    print(f"Model:    {cfg.model.name} (LR: {cfg.model.lr}, Epochs: {cfg.model.epochs})")
    print(f"Batch:    {cfg.dataset.batch_size}")
    print(f"Seed:     {cfg.dataset.seed}")
    print(f"-------------------------\n")
    
    # 2. Setup Data
    train_loader, val_loader, class_names = make_dataloaders(cfg)
    
    # 3. Setup Model
    model = BeverageModelResnet(
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone
    ).to(device)

    # 4. Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.model.lr)

    history = []
    best_acc = 0.0

    start_total = time.time()

    # 5. The Training Loop (Variable epochs from Hydra)
    for epoch in range(1, cfg.model.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Standard PyTorch Step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                val_total += labels.size(0)

        # Calculate averages
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch [{epoch}/{cfg.model.epochs}] - "
              f"Train Acc: {epoch_train_acc:.3f} | Val Acc: {epoch_val_acc:.3f}")

        # Save Checkpoint if accuracy improves
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), cfg.path_model)

        history.append({
            "epoch": epoch,
            "train_acc": epoch_train_acc,
            "val_acc": epoch_val_acc
        })

    total_duration = time.time() - start_total
    print(f"Total Training Time: {total_duration/60:.2f} minutes\n")

    # 6. Final Export
    with open(cfg.path_metrics_train, "w") as f:
        json.dump(history, f, indent=4)
    print("✅ Training Complete. Metrics and Model saved.")

if __name__ == "__main__":
    train()