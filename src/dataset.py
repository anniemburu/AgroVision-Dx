import torch
from torchvision import datasets, transforms
from pathlib import Path

def get_classes(data_dir = "data/train"):
    data_path = Path(data_dir)
    return [d.name for d in data_path.iterdir() if d.is_dir()]

def get_data_loader(data_dir, batch_size=32):
    ## ViT Specific transforms

    # Trining
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Validation
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_set = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_set = datasets.ImageFolder(f"{data_dir}/valid", transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, train_set.classes