import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig
from PIL import Image
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes")

class AlcDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
def get_transforms():
    """Returns the math operations to clean and prep our images."""
    # We normalize to match the structure of ResNet
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        ])
    return train_transform

def robust_loader(path):
        with Image.open(path) as img:
            return img.convert("RGB")

@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def preprocess(cfg: DictConfig):
    """Converts raw images to saved .pt tensors once."""
    print(f"Preprocessing images from {cfg.dataset.path_raw}...")
    
    raw_dataset = datasets.ImageFolder(
        root=cfg.dataset.path_raw, 
        transform=get_transforms(),
        loader=robust_loader
    )
    
    # We load everything into memory to save it
    all_images = []
    all_labels = []
    
    for img, label in raw_dataset:
        all_images.append(img)
        all_labels.append(label)
        
    # Create the directory if it doesn't exist
    processed_path = Path(cfg.dataset.path_processed)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Save as .pt files
    torch.save(torch.stack(all_images), processed_path / "images.pt")
    torch.save(torch.tensor(all_labels), processed_path / "labels.pt")
    # Also save the class names so we don't lose them
    torch.save(raw_dataset.classes, processed_path / "classes.pt")
    
    print(f"✅ Preprocessing complete. Files saved in {processed_path}")

def make_dataloaders(cfg: DictConfig):
    """Loads the pre-processed tensors and splits them."""
    processed_path = Path("data/processed")
    
    # Check if files exist, if not, suggest preprocessing
    if not (processed_path / "images.pt").exists():
        print("❌ Processed data not found. Please run 'python data.py' first.")
        return

    dataset = AlcDataset(
        processed_path / "images.pt", 
        processed_path / "labels.pt"
    )
    class_names = torch.load(processed_path / "classes.pt")

    # Split into Train and Validation
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=cfg.dataset.val_fraction,
        stratify=dataset.labels.tolist(), # Use the pre-loaded labels
        random_state=cfg.dataset.seed
    )
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.dataset.batch_size, shuffle=True,
        num_workers=cfg.dataset.num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg.dataset.batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Dataloaders successfully created.\n")

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    preprocess() # Run this when calling 'python data.py'